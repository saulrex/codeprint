import unittest

from codeprint import clike, logiclines
from codeprint.wrapping_axis import extract_wrapping_examples


class TestWrappingAxis(unittest.TestCase):
    def test_python_wrapped_and_flat(self):
        src = (
            "result = compute(\n"
            "    alpha, beta)\n"
            "x = 1\n"
        )
        ex = extract_wrapping_examples(src, logiclines)
        self.assertEqual(len(ex), 2)
        self.assertEqual(ex[0]["label"], 1)
        self.assertEqual(ex[1]["label"], 0)
        self.assertGreater(ex[0]["n_chars"], ex[1]["n_chars"])

    def test_ts_chain(self):
        src = (
            "const r = foo\n"
            "  .bar()\n"
            "  .baz();\n"
            "const y = 2;\n"
        )
        ex = extract_wrapping_examples(src, clike)
        self.assertEqual([e["label"] for e in ex], [1, 0])

    def test_comments_dont_fake_wrap(self):
        src = (
            "# комментарий\n"
            "x = 1\n"
        )
        ex = extract_wrapping_examples(src, logiclines)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["label"], 0)

    def test_template_literal_skipped(self):
        src = (
            "const s = `a\n"
            "\n"
            "b`;\n"
            "const x = 1;\n"
        )
        ex = extract_wrapping_examples(src, clike)
        # template-литерал не считается «переносом», стейтмент с ним пропущен
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["next_line"], "const x = 1;")

    def test_no_label_leak_in_prev_tail(self):
        # анти-утечки: одна склеенная строка для обоих классов,
        # признак «кончается на ;» не выдаёт ответ
        src = "result = compute(\n    alpha, beta)\nx = 1\n"
        ex = extract_wrapping_examples(src, logiclines)
        self.assertEqual(len(ex[0]["prev_tail"]), 1)
        self.assertEqual(len(ex[1]["prev_tail"]), 1)
        self.assertIn("alpha, beta)", ex[0]["prev_tail"][0])  # склейка
        self.assertTrue(ex[0]["prev_tail"][0].rstrip().endswith(")"))

    def test_examples_fit_encoder(self):
        from codeprint.features import N_NUM, encode_example
        ex = extract_wrapping_examples("x = compute(\n    1, 2)\n", logiclines)
        ids, nums, y = encode_example(ex[0], {})
        self.assertEqual(len(nums), N_NUM)
        self.assertEqual(y, 1)


if __name__ == "__main__":
    unittest.main()
