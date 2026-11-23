"""Show that a decision log is tamper-evident: it replays when untouched and fails after one edited character.

    python scripts/replay_demo.py reports/evidence/example_analysis_log.json
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.harness.replay import replay_analysis


def main(path: str) -> None:
    record = replay_analysis(path)
    print(f"1. Untouched log replays: action={record['action']}, tools={[c['tool_name'] for c in record['plan']]}, evidence={[e['evidence_id'] for e in record['evidence']]}")
    print(f"   states: {' -> '.join(s if isinstance(s, str) else s.get('state', str(s)) for s in record['state_history'])}")
    print(f"   guard violations: {record.get('violations')}, raw draft stored: {'raw_answer' in record}, final answer == raw draft: {record.get('raw_answer') == record['answer']}")
    print(f"   hash:   {record['replay_sha256']}")
    text = Path(path).read_text(encoding="utf-8")
    tampered = text.replace("REJECT_SIGNAL", "WATCH", 1) if "REJECT_SIGNAL" in text else text.replace("0", "1", 1)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "tampered.json"
        copy.write_text(tampered, encoding="utf-8")
        try:
            replay_analysis(copy)
            print("2. Tampered copy replayed: INTEGRITY CHECK DID NOT FIRE")
        except ValueError as exc:
            print(f"2. Tampered copy (one edit) is rejected: {exc}")
    assert json.loads(text)["replay_sha256"] == record["replay_sha256"]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/evidence/example_analysis_log.json")
