import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_sft import validate_file


class TajikSFTValidationTests(unittest.TestCase):
    def write_rows(self, root, rows):
        path = root / "sft.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_accepts_tajik_messages_format(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.write_rows(root, [{
                "messages": [
                    {"role": "user", "content": "Ин калима чӣ маъно дорад?"},
                    {"role": "assistant", "content": "Ин калима маънои зеринро дорад."}
                ]
            }])
            result = validate_file(path, 0.55)
            self.assertEqual(result["examples"], 1)
            self.assertEqual(result.get("low_cyrillic_assistant_answers", 0), 0)

    def test_reports_duplicate_conversations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = {"messages": [
                {"role": "user", "content": "Салом"},
                {"role": "assistant", "content": "Салом, хуш омадед."}
            ]}
            path = self.write_rows(root, [row, row])
            result = validate_file(path, 0.55)
            self.assertEqual(result["duplicate_examples"], 1)

    def test_flags_non_cyrillic_assistant_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.write_rows(root, [{
                "messages": [
                    {"role": "user", "content": "Салом"},
                    {"role": "assistant", "content": "Hello, how are you?"}
                ]
            }])
            result = validate_file(path, 0.55)
            self.assertEqual(result["low_cyrillic_assistant_answers"], 1)

    def test_rejects_unknown_message_role(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.write_rows(root, [{
                "messages": [
                    {"role": "function", "content": "Натиҷа"},
                    {"role": "assistant", "content": "Ҷавоб"}
                ]
            }])
            with self.assertRaises(ValueError):
                validate_file(path, 0.55)


if __name__ == "__main__":
    unittest.main()
