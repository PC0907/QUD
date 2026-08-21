"""Data loading, splitting, and the label taxonomy.

Submodules are imported directly (e.g. `from qud_evasion.data.load import
load_qevasion`) to keep `import qud_evasion` free of heavy dependencies.
"""
from . import taxonomy  # light, safe to expose eagerly
