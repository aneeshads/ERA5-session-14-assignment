"""Builds Session-14-assignment.ipynb from notebook_src.py (the source of truth).

    python3 build_notebook.py                 # write the .ipynb
    python3 build_notebook.py --flat out.py   # also write a flat script (all cells in order) for smoke tests
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SRC = (HERE / "notebook_src.py").read_text()

cells = []
for chunk in re.split(r"^# %%", SRC, flags=re.M):
    if not chunk.strip():
        continue
    first, _, body = chunk.partition("\n")
    if first.strip() == "[markdown]":
        text = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.strip("\n").splitlines())
        cells.append(nbf.v4.new_markdown_cell(text))
    else:
        cells.append(nbf.v4.new_code_cell(body.strip("\n")))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "accelerator": "TPU", "colab": {"provenance": [], "gpuType": "V5E1"}}
out = HERE / "Session-14-assignment.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")

if "--flat" in sys.argv:
    flat = Path(sys.argv[sys.argv.index("--flat") + 1])
    code = [c["source"] for c in cells if c["cell_type"] == "code"]
    flat.write_text("\n\n# ---- cell ----\n".join(code) + "\n")
    print(f"wrote {flat}")
