import unittest

from codeprint.profiles import CompactDecider, KeepDecider, ParagraphDecider
from codeprint.restyler import restyle

SAMPLE = """import os
x = 1
y = 2
def foo():
    a = 1
    b = 2
    return a + b
"""


class TestRestyle(unittest.TestCase):
    def test_result_always_compiles(self):
        for d in (CompactDecider(), ParagraphDecider(), KeepDecider()):
            compile(restyle(SAMPLE, d), "<t>", "exec")

    def test_idempotent(self):
        for d in (CompactDecider(), ParagraphDecider(), KeepDecider()):
            once = restyle(SAMPLE, d)
            self.assertEqual(once, restyle(once, d))

    def test_compact_strips_inner_blanks(self):
        src = "def f():\n    a = 1\n\n    b = 2\n    return a + b\n"
        out = restyle(src, CompactDecider())
        self.assertNotIn("\n\n    b", out)

    def test_paragraph_adds_blank_before_return(self):
        src = "def f():\n    a = 1\n    b = 2\n    return a + b\n"
        out = restyle(src, ParagraphDecider())
        self.assertIn("b = 2\n\n    return", out)

    def test_paragraph_blank_after_assign_run(self):
        src = "def f():\n    a = 1\n    b = 2\n    print(a + b)\n"
        out = restyle(src, ParagraphDecider())
        self.assertIn("b = 2\n\n    print", out)

    def test_decorator_never_split(self):
        src = "def deco(f):\n    return f\n@deco\ndef g():\n    pass\n"
        out = restyle(src, ParagraphDecider())
        self.assertIn("@deco\ndef g():", out)

    def test_else_not_detached(self):
        src = "def f(x):\n    if x:\n        a = 1\n    else:\n        a = 2\n    return a\n"
        out = restyle(src, ParagraphDecider())
        self.assertIn("a = 1\n    else:", out)

    def test_blank_inside_multiline_string_preserved(self):
        src = 's = """line1\n\nline2"""\nx = 1\n'
        out = restyle(src, CompactDecider())
        self.assertIn('"""line1\n\nline2"""', out)

    def test_nonblank_chars_untouched(self):
        src = SAMPLE
        out = restyle(src, ParagraphDecider())
        strip = lambda s: [l for l in s.splitlines() if l.strip()]
        self.assertEqual(strip(src), strip(out))

    def test_keep_normalizes_blank_runs(self):
        src = "x = 1\n\n\n\ny = 2\n"
        out = restyle(src, KeepDecider())
        self.assertEqual(out, "x = 1\n\ny = 2\n")


if __name__ == "__main__":
    unittest.main()
