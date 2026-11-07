import tempfile
import unittest
from pathlib import Path

from codeprint.naming_axis import make_renamer
from codeprint.public_names import (apply_public_map, collect_exports, run,
                                    safety_scan)

VOCAB = ["data", "payload", "entity", "meta", "result", "buffer", "handle"]


class TestPublicNames(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "a.ts").write_text(
            "export const userHelper = () => 1;\n"
            "export const secretKey = 'userHelper';\n", encoding="utf-8")
        (self.tmp / "b.ts").write_text(
            "import { userHelper } from './a';\n"
            "const x = userHelper();\n"
            "const y = obj.userHelper;\n", encoding="utf-8")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_collect_exports(self):
        ex = collect_exports(self.tmp)
        self.assertIn("userHelper", ex)
        self.assertIn("secretKey", ex)

    def test_safety_flags_string_refs(self):
        reasons = safety_scan(self.tmp, {"userHelper", "secretKey"})
        # userHelper упомянут в строке 'userHelper' в a.ts -> небезопасен
        self.assertTrue(reasons["userHelper"])
        self.assertFalse(reasons["secretKey"])

    def test_apply_skips_dot_access(self):
        src = "userHelper();\nobj.userHelper;\n"
        out = apply_public_map(src, {"userHelper": "payloadMeta"})
        self.assertIn("payloadMeta()", out)
        self.assertIn("obj.userHelper", out)

    def test_run_dry_run_writes_nothing(self):
        before = (self.tmp / "b.ts").read_text()
        report = run(self.tmp, make_renamer(VOCAB, "t"), in_place=False)
        self.assertEqual((self.tmp / "b.ts").read_text(), before)
        self.assertIn("map", report)

    def test_run_in_place_consistent(self):
        # secretKey безопасен (строковая ссылка — на userHelper, не на него)
        report = run(self.tmp, make_renamer(VOCAB, "t"), in_place=True)
        new_name = report["map"].get("secretKey")
        self.assertTrue(new_name)
        a = (self.tmp / "a.ts").read_text()
        self.assertIn(new_name, a)
        self.assertNotIn("export const secretKey", a)


if __name__ == "__main__":
    unittest.main()
