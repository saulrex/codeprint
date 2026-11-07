import unittest

from codeprint import clike, logiclines
from codeprint.comments_axis import apply_comments, extract_comments_examples


class TestComments(unittest.TestCase):
    def test_extract_py_trailing_and_own(self):
        src = "x = 1  # хвост\n# своя строка\ny = 2\n"
        ex = extract_comments_examples(src, logiclines)
        self.assertEqual([e["label"] for e in ex], [1, 0])

    def test_extract_ts(self):
        src = "const x = 1; // хвост\n// своя\nconst y = 2;\n"
        ex = extract_comments_examples(src, clike)
        self.assertEqual([e["label"] for e in ex], [1, 0])

    def test_hash_inside_string_not_comment(self):
        src = 'url = "http://x#frag"  # настоящий\n'
        ex = extract_comments_examples(src, logiclines)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["next_line"], "// настоящий")

    def test_apply_moves_long_trailing_up(self):
        src = "const somethingLong = computeValue(argument); // длинный комментарий\n"
        out = apply_comments(src, clike, trailing_max_chars=40)
        lines = out.splitlines()
        self.assertTrue(lines[0].strip().startswith("//"))
        self.assertEqual(lines[1].strip(), "const somethingLong = computeValue(argument);")

    def test_apply_pulls_own_line_to_tail(self):
        src = "x = 1\n# короткий\ny = 2\n"
        out = apply_comments(src, logiclines, trailing_max_chars=40)
        # комментарий прилипает к предыдущей строке, если влезает в порог
        self.assertIn("# короткий", out.splitlines()[0])

    def test_apply_python_compiles(self):
        src = "def f():\n    x = 1  # значение\n    return x\n"
        out = apply_comments(src, logiclines, trailing_max_chars=20)
        compile(out, "<t>", "exec")


if __name__ == "__main__":
    unittest.main()
