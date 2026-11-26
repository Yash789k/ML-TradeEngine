"""Run one bounded proposal through the OpenAI managed agent harness."""

import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel
from typing import Literal


class Proposal(BaseModel):
    option: Literal["A", "B", "C", "D", "E"]
    reason: str


def propose(folder, accuracy, year, horizon):
    options = {"A": 0.1, "B": 0.3, "C": 1.0, "D": 3.0, "E": 10.0}
    prompt = (
        "Select one L2 strength for the next logistic expert-weighting gate. "
        f"The incumbent uses 1. Options: {json.dumps(options)}. The only evidence "
        f"is earlier out-of-time classification accuracy: {json.dumps(accuracy, sort_keys=True)}. "
        "Account for dependence and limited evidence. Supply the option and a brief reason."
    )
    name = f"agent-{year}-h{horizon}.json"
    paths = [folder / name, Path(__file__).with_name("receipts") / name]
    for path in paths:
        if path.is_file():
            cached = json.loads(path.read_text())
            if path == paths[1] and not os.environ.get("CECI2_HOSTED_REVIEW"):
                return dict(
                    cached, replayed=True, exact_input_match=cached["prompt"] == prompt
                )
            if cached["prompt"] == prompt:
                return cached
    if not os.environ.get("CECI2_HOSTED_REVIEW") or not os.environ.get(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "A new agent review requires OPENAI_API_KEY. Recorded reviews are available for the frozen study."
        )
    (folder / (name + ".input")).write_text(prompt)
    client = OpenAI(max_retries=0, timeout=120)
    stream = client.beta.agents.sessions.create(
        agent={
            "model": "gpt-6-sol",
            "instructions": "Review only the supplied historical metrics. Select one permitted research trial.",
        },
        environment={"type": "none"},
        input=prompt,
        output_type=Proposal,
        stream=True,
        spend_control={"limit": 50},
    )
    with stream:
        result = stream.get_final_result()
    choice = result.output_parsed
    receipt = dict(
        service="OpenAI Agents API",
        model="gpt-6-sol",
        prompt=prompt,
        option=choice.option,
        l2=options[choice.option],
        reason=choice.reason,
        session_id=result.session_id,
        turn_id=result.turn_id,
        cap_usd=0.50,
        usage=result.turn.usage.model_dump() if result.turn.usage else None,
    )
    paths[0].write_text(json.dumps(receipt, indent=2))
    return receipt
