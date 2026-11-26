"""Show the recorded five-session decision for one instrument and its next-open fills."""
import argparse
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "site/assets"


def decision(date, symbol):
    tape = json.loads((ASSETS / "simulation.json").read_text())
    if date not in tape["dates"] or symbol not in tape["symbols"]:
        raise ValueError("Choose a decision date and instrument listed in the website controller")
    day = tape["dates"].index(date)
    asset = tape["symbols"].index(symbol)
    detail = json.loads((ASSETS / f"detail-{date[:4]}.json").read_text())
    row = detail["dates"].index(date)
    return {
        "decision_date": date,
        "symbol": symbol,
        "market_state_probabilities": tape["posterior"][day],
        "context": dict(zip(detail["contextNames"], detail["context"][row][asset])),
        "expert_predictions": [
            {"expert": name, "weight": weight, "down_flat_up": probabilities}
            for name, weight, probabilities in zip(
                detail["experts"], detail["weights"][row][asset], detail["probabilities"][row][asset]
            )
        ],
        "combined": {name: tape[name][day][asset] for name in ("down", "flat", "up")},
        "next_open": tape["entryDates"][day],
        "dynamic_policy_orders": [
            order for order in tape["orders"] if order["signal_date"] == date and order["symbol"] == symbol
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date")
    parser.add_argument("--symbol", default="AAPL", type=str.upper)
    args = parser.parse_args()
    try:
        print(json.dumps(decision(args.date, args.symbol), indent=2))
    except ValueError as error:
        parser.error(str(error))
