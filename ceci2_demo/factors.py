"""Published US factor-theme returns from Jensen, Kelly and Pedersen."""

import subprocess
import numpy as np
import pandas as pd

URL = (
    "https://jkpfactors-data.s3.amazonaws.com/public/"
    "%5Busa%5D_%5Ball_themes%5D_%5Bdaily%5D_%5Bvw_cap%5D.zip"
)


def signals(folder, dates):
    path = folder / "jkp.zip"
    if not path.exists():
        tmp = path.with_suffix(".part")
        subprocess.run(
            ["curl", "-fsSL", "--retry", "3", URL, "-o", str(tmp)], check=True
        )
        tmp.replace(path)
    frame = pd.read_csv(path, parse_dates=["date"])
    returns = frame.pivot(index="date", columns="name", values="ret").sort_index()
    selected = returns.loc[dates.min() : dates.max()]
    if selected.isna().any().any() or not selected.index.equals(dates):
        raise ValueError("JKP daily factor coverage differs from the market calendar")
    # Historical factor releases can be revised; these are retrospective features.
    trailing = ((1 + returns).rolling(21).apply(np.prod, raw=True) - 1).shift(21)
    return trailing.reindex(dates).add_prefix("jkp_"), selected
