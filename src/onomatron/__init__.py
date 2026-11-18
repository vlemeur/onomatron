"""onomatron: character-level language models for French literature names."""

import copy

import plotly.express as px
import plotly.io as pio

# Default look for every plotly figure in the notebooks: the clean white theme
# with the Set2 qualitative palette. Set once here so any `import onomatron...`
# applies it, including figures built directly in the notebooks. deepcopy +
# override: nesting template='simple_white' inside a new Template loses the
# custom colorway at render time.
_template = copy.deepcopy(pio.templates["simple_white"])
_template.layout.colorway = list(px.colors.qualitative.Set2)
pio.templates["onomatron"] = _template
pio.templates.default = "onomatron"
