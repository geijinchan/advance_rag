"""Golden evaluation set — RAGAS-style Q&A pairs over the demo corpus.

Each case pins down *what a correct answer must contain* (fact keywords that
must appear in the answer text) and *where the evidence must come from*
(any-of expected source documents). Route-aware cases check the agentic
behaviour itself (honest fallback for out-of-scope, chitchat handling).

This is the regression harness for the service: every retrieval- or
answer-composition change is validated against the whole set via
``POST /eval`` or ``scripts/eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvalCase:
    id: str
    question: str
    expects: Literal["grounded", "fallback", "chitchat"]
    # At least ONE keyword/regex must appear in the answer (case-insensitive).
    answer_must_match_any: list[str] = field(default_factory=list)
    # At least ONE of these doc names must appear in the citations.
    docs_any: list[str] = field(default_factory=list)
    category: str = "factoid"      # factoid | lookup | table | behaviour | robustness
    note: str = ""


GOLDEN_SET: list[EvalCase] = [
    EvalCase(
        id="warranty-x200-standard",
        question="What is the standard warranty on the X200?",
        expects="grounded",
        answer_must_match_any=[r"24\s*months"],
        docs_any=["x200_manual.pdf", "x200_controller_manual.md", "warranty_policy"],
        category="table",
        note="warranty matrix row; guards against OCR chart-garble answers",
    ),
    EvalCase(
        id="warranty-x200-extended",
        question="How long is the extended warranty for the X200 and what is the registration window?",
        expects="grounded",
        answer_must_match_any=[r"36\s*months"],
        docs_any=["x200_manual.pdf", "x200_controller_manual.md", "warranty_policy"],
        category="table",
    ),
    EvalCase(
        id="s200-accuracy",
        question="What is the accuracy of the S200 sensor?",
        expects="grounded",
        answer_must_match_any=[r"0\.3\s*°?C", r"\+/-\s*0\.3"],
        docs_any=["s200_sensor_datasheet.md"],
        category="factoid",
    ),
    EvalCase(
        id="led-2hz",
        question="What does a blinking red LED at 2 Hz indicate?",
        expects="grounded",
        answer_must_match_any=["overtemperature"],
        docs_any=["troubleshooting_flowchart.md"],
        category="table",
        note="guards against wrong table-row pairing (the 'Off' row regression)",
    ),
    EvalCase(
        id="rma-evaluation-time",
        question="What is the RMA processing time?",
        expects="grounded",
        answer_must_match_any=[r"48\s*hours"],
        docs_any=["rma_faq.txt"],
        category="robustness",
        note="paraphrase gap: question says 'processing time', corpus says 'evaluation takes'",
    ),
    EvalCase(
        id="modbus-fix",
        question="Which firmware version fixed the MODBUS-TCP timeout bug?",
        expects="grounded",
        answer_must_match_any=["3\\.2\\.1", "v3.2.1"],
        docs_any=["firmware_release_notes", "x200_manual.pdf", "x200_controller_manual.md"],
        category="lookup",
    ),
    EvalCase(
        id="s200-temp-range",
        question="What is the operating temperature range of the S200 sensor?",
        expects="grounded",
        answer_must_match_any=[r"-40", "125"],
        docs_any=["s200_sensor_datasheet.md"],
        category="factoid",
    ),
    EvalCase(
        id="clearance",
        question="What clearance is required around the controller for cooling?",
        expects="grounded",
        answer_must_match_any=[r"100\s*mm"],
        docs_any=["deployment_checklist.md", "safety_compliance_guide.md", "troubleshooting_flowchart.md"],
        category="factoid",
    ),
    EvalCase(
        id="mqtt-port",
        question="Which port must production MQTT connections use?",
        expects="grounded",
        answer_must_match_any=["8883"],
        docs_any=["network_integration_guide.md"],
        category="factoid",
    ),
    EvalCase(
        id="e103-meaning",
        question="What does error code E103 mean?",
        expects="grounded",
        answer_must_match_any=["temperature", "overtemp"],
        docs_any=["troubleshooting_flowchart.md"],
        category="table",
    ),
    EvalCase(
        id="x100-ip-rating",
        question="What is the ingress protection rating of the X100 controller?",
        expects="grounded",
        answer_must_match_any=["IP54"],
        docs_any=["x100_controller_manual.md", "warranty_policy.md", "rma_faq.txt"],
        category="factoid",
    ),
    EvalCase(
        id="compare-x200-x300-warranty",
        question="Compare the standard warranty of the X200 and the X300.",
        expects="grounded",
        answer_must_match_any=[r"(?s)(?=.*24[-\s]?months?)(?=.*36[-\s]?months?)"],
        docs_any=["x200_manual.pdf", "x200_controller_manual.md", "x300_controller_manual.md", "warranty_policy"],
        category="multihop",
        note="comparative question — exercises query decomposition; both entities' facts must appear",
    ),
    EvalCase(
        id="compare-x100-x300-warranty",
        question="What is the difference between the warranty on the X100 and the warranty on the X300?",
        expects="grounded",
        answer_must_match_any=[r"(?s)(?=.*12[-\s]?months?)(?=.*36[-\s]?months?)"],
        docs_any=["x100_controller_manual.md", "x300_controller_manual.md"],
        category="multihop",
        note="second decomposition case: 12 vs 36 months standard warranty",
    ),
    EvalCase(
        id="out-of-scope-worldcup",
        question="Who won the 2022 FIFA World Cup?",
        expects="fallback",
        answer_must_match_any=["couldn't find", "not", "out-of-scope", "out of scope"],
        category="behaviour",
        note="honest fallback — must NOT fabricate an answer",
    ),
    EvalCase(
        id="out-of-scope-swallow",
        question="What is the airspeed velocity of an unladen swallow?",
        expects="fallback",
        answer_must_match_any=["couldn't find", "not", "out-of-scope", "out of scope"],
        category="behaviour",
    ),
    EvalCase(
        id="chitchat-greeting",
        question="hello there!",
        expects="chitchat",
        answer_must_match_any=["hello", "hi", "assistant"],
        category="behaviour",
    ),
]
