import json
from pathlib import Path

from common import md
import part_intro
import part1
import part2
import part3a
import part3b
import part4
import part5
import part6_10
import part11_13
import part14
import part15

cells = []
for mod in (
    part_intro,
    part1,
    part2,
    part3a,
    part3b,
    part4,
    part5,
    part6_10,
    part11_13,
    part14,
    part15,
):
    cells.extend(mod.cells())

cells.append(
    md(
        """
# Appendix — files in this repo

| Path | What |
|------|------|
| `rag_coding_prep.ipynb` | this notebook |
| `rag/` | Part 14 package |
| `data/sample_docs/` | tiny corpus |
| `tests/test_rag_components.py` | component tests (no FastEmbed required) |
| `rag_oa_practice.ipynb` | older drill notebook (stdlib TODOs) |

```bash
python -m pytest tests/test_rag_components.py -q
python -m rag.main          # needs fastembed
```
"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": [{"id": f"c{i:04d}", **cell} for i, cell in enumerate(cells)],
}

out = Path(__file__).resolve().parents[1] / "rag_coding_prep.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
