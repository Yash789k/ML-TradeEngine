"""Export the per-model decisions shown in the site's comparison inspector."""
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2]
studies = root / "data/ceci2-demo"
def export(head_file, router_file, filename):
    experts = pd.read_csv(studies/head_file)
    router = pd.read_csv(studies/router_file)
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
    (root/"site/assets"/filename).write_text(json.dumps(payload,separators=(",",":"),allow_nan=False))


if __name__ == "__main__":
    export("joint-v1/experts.csv.gz", "rl-router-v1/forecasts.csv.gz", "model-trace.json")
    export("colab-t4-v1/joint-experts/experts.csv.gz", "colab-t4-v1/expert-selector/forecasts.csv.gz", "gpu-trace.json")
