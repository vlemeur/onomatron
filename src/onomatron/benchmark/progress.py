"""Single-bar progress display for the whole benchmark run.

One tqdm bar for everything (all models x all trials, plus the eval runs),
with the current context (model / trial / step / best val NLL) in the bar's
postfix — instead of stacking several bars that fight over the terminal.
Readable both live in a terminal and in a redirected log file. The bar writes
to stdout so `> bench.log 2>&1` keeps logs (stderr) clean of bar frames.
"""

import logging
import sys
import time
from types import TracebackType

from tqdm import tqdm

logger = logging.getLogger(__name__)


class BenchProgress:
    """Progress display for the entire benchmark.

    - TTY (real terminal): an in-place tqdm bar (one line that overwrites
      itself, no scrolling) with the context in the postfix.
    - non-TTY (pipe, IDE panel, redirected log): no bar frames at all —
      a throttled one-line INFO log carries the same information, so the
      log stays readable.
    """

    def __init__(
        self, total: int, desc: str = "bench", log_every_s: float = 5.0
    ) -> None:
        self.tty = sys.stderr.isatty()
        self.log_every_s = log_every_s
        self._last_log = 0.0
        self.bar = tqdm(
            total=total,
            desc=desc,
            unit="run",
            file=sys.stderr,  # same stream as logging (routed via tqdm.write)
            disable=not self.tty,  # bars only make sense on a real terminal
            mininterval=0.5,
            position=0,  # stays on the first line; train bars stack below
            leave=True,
        )
        self.model = "-"
        self.trial = 0
        self.n_trials = 0
        self.step = 0
        self.steps = 0
        self.best: float | None = None

    def __enter__(self) -> BenchProgress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def start_model(self, model: str, n_trials: int, steps: int) -> None:
        """New model (study) starts: reset the per-model context."""
        self.model = model
        self.trial = 0
        self.n_trials = n_trials
        self.step = 0
        self.steps = steps
        self.best = None
        self._refresh()

    def set_step(self, step: int, best: float | None = None) -> None:
        """Update the step counter (called from train's on_eval hook)."""
        self.step = step
        if best is not None:
            self.best = best
        self._refresh()
        self._maybe_log()

    def trial_done(self, best: float | None = None) -> None:
        """One run (trial or eval) finished: advance the bar by one."""
        self.trial += 1
        if best is not None:
            self.best = best
        self.bar.update(1)
        self._refresh()
        self._maybe_log()

    def _maybe_log(self) -> None:
        """Non-TTY fallback: one throttled INFO line with the same context."""
        if self.tty:
            return
        now = time.monotonic()
        if now - self._last_log < self.log_every_s:
            return
        self._last_log = now
        best = f"{self.best:.4f}" if self.best is not None else "-"
        logger.info(
            "progress %d/%d model=%s trial=%d/%d step=%d/%d best=%s",
            self.bar.n,
            self.bar.total,
            self.model,
            self.trial,
            self.n_trials,
            self.step,
            self.steps,
            best,
        )

    def _refresh(self) -> None:
        best = f"{self.best:.4f}" if self.best is not None else "-"
        # refresh=False: let tqdm's mininterval decide when to redraw,
        # instead of forcing a write on every step/trial.
        self.bar.set_postfix_str(
            f"model={self.model} trial={self.trial}/{self.n_trials} "
            f"step={self.step}/{self.steps} best={best}",
            refresh=False,
        )

    def close(self) -> None:
        self.bar.close()
