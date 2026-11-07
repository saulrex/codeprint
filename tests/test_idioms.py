import unittest

from codeprint import clike, logiclines
from codeprint.idioms_axis import apply_idioms, extract_idioms_examples

TS_IFELSE = """function f(x) {
  if (x > 0) {
    return 'pos';
  }
  return 'neg';
}
"""

TS_TERNARY = """function f(x) {
  return x > 0 ? 'pos' : 'neg';
}
"""

PY_IFELSE = """def f(x):
    if x > 0:
        return 'pos'
    return 'neg'
"""


class TestIdioms(unittest.TestCase):
    def test_detect_ts_both_forms(self):
        self.assertEqual(extract_idioms_examples(TS_IFELSE, clike)[0]["label"], 0)
        self.assertEqual(extract_idioms_examples(TS_TERNARY, clike)[0]["label"], 1)

    def test_detect_py(self):
        ex = extract_idioms_examples(PY_IFELSE, logiclines)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["label"], 0)

    def test_optional_chaining_not_ternary(self):
        src = "function f(x) {\n  return x?.value;\n}\n"
        self.assertEqual(extract_idioms_examples(src, clike), [])

    def test_apply_to_ternary_ts(self):
        out = apply_idioms(TS_IFELSE, clike, ternary_max_chars=100)
        self.assertIn("return x > 0 ? 'pos' : 'neg';", out)

    def test_apply_to_ifelse_when_long(self):
        out = apply_idioms(TS_TERNARY, clike, ternary_max_chars=5)
        self.assertIn("if (x > 0) {", out)
        self.assertIn("return 'neg';", out)
        self.assertNotIn("else", out)  # авторская форма — ранний return

    def test_apply_python_roundtrip(self):
        out = apply_idioms(PY_IFELSE, logiclines, ternary_max_chars=100)
        self.assertIn("return 'pos' if x > 0 else 'neg'", out)
        compile(out, "<t>", "exec")
        back = apply_idioms(out, logiclines, ternary_max_chars=5)
        compile(back, "<t>", "exec")
        self.assertIn("if x > 0:", back)

    def test_threshold_keeps_matching(self):
        # конструкция уже в нужной форме — не трогаем
        out = apply_idioms(TS_TERNARY, clike, ternary_max_chars=100)
        self.assertEqual(out, TS_TERNARY)


if __name__ == "__main__":
    unittest.main()
