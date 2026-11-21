import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_corpus import normalize_text, prepare, script_stats, stable_split


class TajikCorpusTests(unittest.TestCase):
    def test_nfc_and_whitespace_normalization(self):
        self.assertEqual(normalize_text("  Салом\u00a0дунё!  \r\n\t "), "Салом дунё!")

    def test_tajik_letters_are_preserved(self):
        value = "Ғғ Ӣӣ Ққ Ӯӯ Ҳҳ Ҷҷ"
        self.assertEqual(normalize_text(value), value)
        stats = script_stats(value)
        self.assertEqual(stats["tajik_marker_letters"], 12)

    def test_script_stats(self):
        stats = script_stats("Тоҷикистон Tajik Հայաստան")
        self.assertGreater(stats["cyrillic_ratio"], 0)
        self.assertGreater(stats["latin_ratio"], 0)

    def test_split_is_deterministic(self):
        a = stable_split("document-42", "test-seed", 5, 5)
        b = stable_split("document-42", "test-seed", 5, 5)
        self.assertEqual(a, b)

    def test_prepare_deduplicates_and_writes_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.jsonl"
            output_dir = root / "out"
            text = ("Тоҷикистон кишвари соҳибистиқлол мебошад. " * 8).strip()
            rows = [
                {"lang": "tg", "text": text, "id": "one"},
                {"lang": "tg", "text": text, "id": "duplicate"},
                {"lang": "ru", "text": "Это русский текст. " * 20, "id": "two"}
            ]
            input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            result = prepare([input_path], output_dir, min_alpha_chars=50)
            self.assertEqual(result["counts"]["documents_kept"], 1)
            self.assertEqual(result["counts"]["rejected"]["exact_duplicate"], 1)
            self.assertEqual(result["counts"]["rejected"]["language_missing_or_not_tajik"], 1)
            self.assertTrue((output_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
