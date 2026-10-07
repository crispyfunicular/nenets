"""Pandas 3 + datasets audiofolder: file_name must be Arrow `string`, not `large_string`."""

import pandas as pd


def enable():
    if hasattr(pd.options, "future"):
        try:
            pd.options.future.infer_string = False
        except Exception:
            pass
