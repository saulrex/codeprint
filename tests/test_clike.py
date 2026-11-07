import unittest

from codeprint import clike
from codeprint.harvest import extract_blankline_examples


class TestClikeScanner(unittest.TestCase):
    def test_labels_012(self):
        src = (
            "const a = 1;\n"
            "const b = 2;\n"   # граница 1: 0 пустых
            "\n"
            "const c = 3;\n"   # граница 2: 1 пустая
            "\n\n\n"
            "const d = 4;\n"   # граница 3: 3 пустые -> label 2 (кап)
        )
        ex = extract_blankline_examples(src, clike)
        self.assertEqual([e["label"] for e in ex], [0, 1, 2])
        self.assertEqual(ex[2]["next_line"], "const d = 4;")
        self.assertTrue(all(e["lang"] == "clike" for e in ex))

    def test_for_loop_semicolons_not_split(self):
        src = (
            "for (let i = 0; i < 3; i++) {\n"
            "  console.log(i);\n"
            "}\n"
        )
        ex = extract_blankline_examples(src, clike)
        # две границы: после '{' и после console.log — и никаких ложных от ';' в for
        self.assertEqual(len(ex), 2)
        self.assertEqual(ex[0]["next_line"], "  console.log(i);")
        self.assertEqual(ex[1]["next_line"], "}")

    def test_method_chain_single_boundary(self):
        src = (
            "const r = foo\n"
            "  .bar()\n"
            "  .baz();\n"
            "\n"
            "const y = 2;\n"
        )
        ex = extract_blankline_examples(src, clike)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["label"], 1)
        self.assertEqual(ex[0]["next_line"], "const y = 2;")

    def test_template_literal_blanks_are_content(self):
        src = (
            "const s = `line1\n"
            "\n"
            "line2`;\n"
            "const x = 1;\n"
        )
        ex = extract_blankline_examples(src, clike)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["label"], 0)
        self.assertEqual(ex[0]["next_line"], "const x = 1;")

    def test_template_interpolation(self):
        src = (
            "const s = `sum: ${a + b}`;\n"
            "\n"
            "const x = 1;\n"
        )
        ex = extract_blankline_examples(src, clike)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["label"], 1)

    def test_block_comment_not_a_boundary(self):
        src = (
            "/* multi\n"
            " * comment\n"
            " */\n"
            "const a = 1;\n"
        )
        ex = extract_blankline_examples(src, clike)
        self.assertEqual(ex, [])  # до const нет логического конца — нет границы

    def test_brace_in_string_ignored(self):
        src = (
            'const s = "a } b";\n'
            "const x = 1;\n"
        )
        ex = extract_blankline_examples(src, clike)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["label"], 0)

    def test_interface_brace_is_boundary(self):
        src = (
            "interface User {\n"
            "  id: number;\n"
            "}\n"
        )
        ex = extract_blankline_examples(src, clike)
        self.assertEqual(len(ex), 2)


if __name__ == "__main__":
    unittest.main()
