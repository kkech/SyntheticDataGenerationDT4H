"""
Re-apply the CURRENT sentinel decode to existing synthetic CSVs, in place.

    python redecode.py

Use case: the decode rule changed after a generation run. The evaluate
step already scores existing files as-if-decoded (it re-decodes in
memory), so the published numbers describe the corrected data -- but the
files on disk still physically contain what the old rule let through
(gap-region values below a column's observed minimum, fractional or
zero NYHA classes). Because the decode is idempotent, rewriting the
files through it makes disk match evaluation EXACTLY, row for row --
unlike regenerate.py, which draws a fresh sample and therefore produces
different (if statistically equivalent) rows from the ones scored.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.steps.generate.step import GenerateStep  # noqa: E402


def main() -> None:
    import pandas as pd

    config = PipelineConfig()
    files = sorted(glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv")))
    if not files:
        print(f"No DT4H_Synthetic_*.csv found in {config.step_dir('generate')}.")
        return

    for path in files:
        df = pd.read_csv(path, low_memory=False)
        df, decoded = GenerateStep._decode_numeric_missing(df, config)
        n_cells = sum(decoded.values())
        if n_cells:
            df.to_csv(path, index=False)
            print(f"✅ {os.path.basename(path)}: {n_cells} cell(s) re-decoded to null "
                  f"across {len(decoded)} column(s); file rewritten.")
        else:
            print(f"✓  {os.path.basename(path)}: already clean, unchanged.")


if __name__ == "__main__":
    main()
