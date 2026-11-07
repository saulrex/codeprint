import json
import re
import tempfile
import unittest
from pathlib import Path

from codeprint import clike, logiclines
from codeprint.composer import Composer
from codeprint.wrapping_axis import apply_wrapping, wrap_statement


class TestWrapApply(unittest.TestCase):
    def test_wrap_long_call(self):
        out = wrap_statement("  const r = fetchData(alpha, beta, gamma);", 2)
        self.assertEqual(out[0], "  const r = fetchData(")
        self.assertIn("    alpha,", out)
        self.assertEqual(out[-1], "  );")

    def test_wrap_no_parens_returns_none(self):
        self.assertIsNone(wrap_statement("x = 5", 2))

    def test_never_wrap_control_flow(self):
        # if/while/return — это НЕ список аргументов: trailing comma там
        # syntax error (JS) или кортеж-бомба (Python)
        self.assertIsNone(wrap_statement("if (alpha || beta || gamma) {", 2))
        self.assertIsNone(wrap_statement("while (alpha && beta) {", 2))
        self.assertIsNone(wrap_statement("return (alpha || beta)", 2))
        self.assertIsNone(wrap_statement("if (alpha || beta):", 2))  # python
        # а вызовы — можно
        self.assertIsNotNone(wrap_statement("foo(alpha, beta)", 2))
        self.assertIsNotNone(wrap_statement("items[0](alpha, beta)", 2))
        self.assertIsNotNone(wrap_statement("factory()(alpha, beta)", 2))

    def test_wrap_respects_strings_with_commas(self):
        out = wrap_statement("foo('a,b', c);", 2)
        self.assertIn("  'a,b',", out)

    def test_apply_wrap_then_join_roundtrip(self):
        src = "const r = fetchData(alpha, beta, gamma);\n"
        wrapped = apply_wrapping(src, clike, lambda ex: 1)
        self.assertIn("alpha,", wrapped)
        joined = apply_wrapping(wrapped, clike, lambda ex: 0)
        self.assertEqual(joined, src)  # wrap -> join = точный roundtrip

    def test_join_skipped_with_comment_inside(self):
        src = "const r = fetchData(\n  // важный коммент\n  alpha);\n"
        out = apply_wrapping(src, clike, lambda ex: 0)
        self.assertIn("важный коммент", out)
        self.assertEqual(len([l for l in out.splitlines() if l.strip()]), 3)

    def test_python_compiles_after_wrap(self):
        src = "result = compute_something(alpha_argument, beta_argument, gamma_argument, delta)\n"
        out = apply_wrapping(src, logiclines, lambda ex: 1)
        compile(out, "<t>", "exec")

    def test_idempotent(self):
        src = "const r = fetchData(alpha, beta, gamma);\n"
        once = apply_wrapping(src, clike, lambda ex: 1)
        twice = apply_wrapping(once, clike, lambda ex: 1)
        self.assertEqual(once, twice)


class TestComposer(unittest.TestCase):
    def test_imports_only_profile(self):
        profile = {"axes": {"imports": {"scheme": "flat_alpha"}}}
        c = Composer(profile)
        src = ("import z from 'z';\nimport a from 'a';\nimport m from 'm';\n"
               "\nconst x = 1;\n")
        out = c.apply(src, ".ts")
        self.assertTrue(out.startswith("import a from 'a';"))
        self.assertEqual(out, c.apply(out, ".ts"))  # идемпотентно

    def test_unknown_ext_passthrough(self):
        c = Composer({"axes": {}})
        self.assertEqual(c.apply("x = 1\n", ".txt"), "x = 1\n")
    def test_idioms_axis(self):
        profile = {"axes": {"idioms": {"ternary_max_chars": 100}}}
        c = Composer(profile)
        src = ("function f(x) {\n  if (x > 0) {\n    return 'pos';\n  }\n"
               "  return 'neg';\n}\n")
        out = c.apply(src, ".ts")
        self.assertIn("return x > 0 ? 'pos' : 'neg';", out)
        self.assertEqual(out, c.apply(out, ".ts"))  # идемпотентно

    def test_comments_axis(self):
        profile = {"axes": {"comments": {"trailing_max_chars": 30}}}
        c = Composer(profile)
        src = "const somethingLong = computeValue(); // длинный комментарий\n"
        out = c.apply(src, ".ts")
        self.assertTrue(out.splitlines()[0].strip().startswith("//"))
    def test_structure_axis(self):
        profile = {"axes": {"structure": {"scheme": "alpha"}}}
        c = Composer(profile)
        src = ("class S {\n  zeta() {\n    return 1;\n  }\n"
               "  alpha() {\n    return 2;\n  }\n}\n")
        out = c.apply(src, ".ts")
        self.assertLess(out.index("alpha()"), out.index("zeta()"))
        self.assertEqual(out, c.apply(out, ".ts"))  # идемпотентно


if __name__ == "__main__":
    unittest.main()
