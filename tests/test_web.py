from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.llm.web import make_server, ui_payload


def _result(**over):
    base = {"question": "q", "as_of_date": "2026-09-10", "action": "INVESTIGATE", "plan": [{"tool_name": "run_backtest", "arguments": {}}],
            "evidence": [{"evidence_id": "E1", "source_tool": "check_data_quality", "values": {"warnings": []}}],
            "raw_answer": "Status: INVESTIGATE. E1", "answer": "Status: INVESTIGATE. E1", "violations": []}
    return {**base, **over}


class PayloadTests(unittest.TestCase):
    def test_clean_answer_splits_status_and_passes_all_checks(self) -> None:
        p = ui_payload(_result())
        self.assertEqual((p["status"], p["answer"], p["guard_replaced"]), ("INVESTIGATE", "E1", False))
        self.assertTrue(all(c["ok"] for c in p["checks"]))
        self.assertIsNone(p["raw_answer"])

    def test_guard_replacement_is_visible(self) -> None:
        p = ui_payload(_result(raw_answer="Status: WATCH. 99.9%", violations=["numeric_hallucination"]))
        self.assertTrue(p["guard_replaced"])
        self.assertEqual(p["raw_answer"], "Status: WATCH. 99.9%")
        self.assertFalse(next(c for c in p["checks"] if c["label"].startswith("Every number"))["ok"])

    def test_blocked_request_has_no_checks(self) -> None:
        p = ui_payload(_result(action="ANALYSIS_UNSAFE", raw_answer=None, plan=[], evidence=[], answer="Status: ANALYSIS_UNSAFE. No data."))
        self.assertEqual((p["checks"], p["guard_replaced"], p["status"]), ([], False, "ANALYSIS_UNSAFE"))


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = make_server(lambda q: {"echo": q}, port=0)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _post(self, data: bytes) -> tuple[int, dict]:
        try:
            with urlopen(Request(f"{self.base}/api/ask", data, {"Content-Type": "application/json"})) as r:
                return r.status, json.loads(r.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_page_samples_and_ask(self) -> None:
        self.assertIn(b"BIST research assistant", urlopen(self.base + "/").read())
        self.assertEqual(len(json.loads(urlopen(self.base + "/api/samples").read())), 5)
        self.assertEqual(self._post(json.dumps({"question": " hi "}).encode()), (200, {"echo": "hi"}))

    def test_bad_requests_are_rejected(self) -> None:
        self.assertEqual(self._post(b"not json")[0], 400)
        self.assertEqual(self._post(json.dumps({"question": "  "}).encode())[0], 400)
        self.assertEqual(self._post(json.dumps({"question": "x" * 1600}).encode())[0], 400)


if __name__ == "__main__":
    unittest.main()
