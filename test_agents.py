import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agents


def fake_ask(system, user, max_tokens=1200):
    if "Intake Agent" in system:
        return {"language": "en", "symptoms": ["weakness"], "duration": "", "guesses": [],
                "topic": "weakness", "emergency_suspected": "", "needs_doctor": False}
    if "Advice Agent" in system:
        return {
            "understood": "You feel weak.",
            "relevant": "These symptoms can be associated with several causes, including dehydration.",
            "to_check": ["Blood pressure"],
            "do_now": ["Rest.", "Drink fluids."],
            "warning_signs": ["fainting", "chest pain", "confusion"],
            "next_step": "See a healthcare professional if this is not better in 48 hours.",
        }
    if "Safety Checker" in system:
        return {"pass": True, "issues": []}
    return {"ok": True}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "referrals.csv"
        self.queue = patch.object(agents, "QUEUE_PATH", self.path)
        self.ask = patch.object(agents, "ask_json", side_effect=fake_ask)
        self.queue.start()
        self.ask.start()
        self.graph = agents.build_graph()

    def tearDown(self):
        self.ask.stop()
        self.queue.stop()
        self.tmp.cleanup()

    def _run(self, **kwargs):
        base = {"text": "I feel weak", "age_group": "adult", "lang_pref": "en",
                "measures": {}, "trace": [], "city": ""}
        base.update(kwargs)
        return self.graph.invoke(base)

    def test_retired_model_is_not_used(self):
        self.assertEqual(agents.resolve_model("llama-3.3-70b-versatile"), agents.PRIMARY)
        self.assertEqual(agents.resolve_model(""), agents.PRIMARY)
        self.assertEqual(agents.resolve_model("openai/gpt-oss-20b"), "openai/gpt-oss-20b")
        self.assertNotEqual(agents.PRIMARY, "llama-3.3-70b-versatile")

    def test_facility_lookup_matches_city(self):
        found = agents.lookup_facilities("Lahore", "yellow", "fever")
        self.assertEqual(found["chosen"], "Mayo Hospital")
        self.assertIn("directory", found["note"].lower())
        unknown = agents.lookup_facilities("Gilgit", "red", "fever")
        self.assertIn("not in the directory", unknown["note"])
        self.assertTrue(unknown["chosen"])

    def test_followup_when_duration_and_city_missing(self):
        out = self._run(text="I feel weak")
        self.assertEqual(out["kind"], "clarify")
        self.assertEqual([q["id"] for q in out["questions"]], ["duration", "city"])
        self.assertEqual(agents.list_cases(), [])

    def test_followup_skips_duration_already_in_text(self):
        out = self._run(text="Mujhe subah se chakkar aa rahe hain aur bohat kamzori hai")
        self.assertEqual(out["kind"], "clarify")
        self.assertEqual([q["id"] for q in out["questions"]], ["city"])

    def test_followup_not_repeated(self):
        out = self._run(text="I feel weak", city="Lahore", duration_hint="2 days", followup_done=True)
        self.assertEqual(out["kind"], "guidance")
        self.assertEqual(out["source"], "groq")
        self.assertEqual(out["facility"]["chosen"], "Mayo Hospital")
        self.assertIn("not been clinically reviewed", out["report"])
        rows = agents.list_cases()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "new")
        self.assertEqual(rows[0]["city"], "Lahore")
        agents.update_status(rows[0]["id"], "seen")
        self.assertEqual(agents.list_cases()[0]["status"], "seen")

    def test_emergency_skips_questions_and_is_logged(self):
        out = self._run(text="Mujhe seene mein dard hai aur saans lene mein mushkil ho rahi hai")
        self.assertEqual(out["kind"], "emergency")
        self.assertNotIn("questions", out)
        self.assertTrue(out["facility"]["chosen"])
        self.assertEqual(len(agents.list_cases()), 1)
        self.assertEqual(agents.list_cases()[0]["risk"], "red")


if __name__ == "__main__":
    unittest.main()
