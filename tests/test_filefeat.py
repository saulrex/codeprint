import unittest

from codeprint import clike, logiclines
from codeprint.filefeat_axis import FEATURE_KEYS, extract_filefeat

PY_SRC = """import os

def foo(a, b):
    x = 1
    y = 2

    return x + y

class Bar:
    def m(self):
        for i in range(10):
            print(i)
"""


class TestFilefeat(unittest.TestCase):
    def test_feature_vector_complete(self):
        ex = extract_filefeat(PY_SRC, logiclines)
        self.assertEqual(len(ex), 1)
        f = ex[0]["features"]
        self.assertEqual(list(f.keys()), FEATURE_KEYS)
        self.assertTrue(all(isinstance(v, float) for v in f.values()))
        self.assertGreater(f["p_blank_return"], 0.9)  # пустая перед return

    def test_poor_file_yields_nothing(self):
        self.assertEqual(extract_filefeat("x = 1\n", logiclines), [])

    def test_ts_file(self):
        src = ("import a from 'a';\nimport b from 'b';\nimport c from 'c';\n"
               "\nconst x = 1;\n\nconst y = 2;\n\nfunction f() {\n"
               "  return x;\n}\n\nconst z = 3;\n\nconst w = 4;\n")
        ex = extract_filefeat(src, clike)
        self.assertEqual(len(ex), 1)
        self.assertGreater(ex[0]["features"]["blank_rate"], 0.3)


if __name__ == "__main__":
    unittest.main()
