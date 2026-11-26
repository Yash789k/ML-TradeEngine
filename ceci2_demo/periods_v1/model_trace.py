"""Export the per-model decisions shown in the site's comparison inspector."""
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2]
studies = root / "data/ceci2-demo"
experts = pd.read_csv(studies/"joint-v1/experts.csv.gz")
router = pd.read_csv(studies/"rl-router-v1/forecasts.csv.gz")
dates = sorted(experts.decision.unique())
symbols = sorted(experts.symbol.unique())
index = pd.MultiIndex.from_product([dates, symbols], names=["decision","symbol"])
payload = {"dates": dates, "symbols": symbols, "heads": {}, "routers": {}}
for name, group in experts.groupby("expert"):
    frame = group.set_index(["decision","symbol"]).loc[index, ["weight","down","flat","up"]]
    payload["heads"][name] = frame.round(5).values.tolist()
for name, group in router.groupby("model"):
    frame = group.set_index(["decision","symbol"]).loc[index]
    names = sorted(frame.selected_expert.unique())
    payload["routers"][name] = {"experts": names,
        "selected": frame.selected_expert.map(names.index).tolist(),
        "forecast": frame[["down","flat","up"]].round(5).values.tolist()}
(root/"site/assets/model-trace.json").write_text(json.dumps(payload,separators=(",",":"),allow_nan=False))
