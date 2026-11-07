import unittest

from codeprint import clike, logiclines
from codeprint.structure_axis import (apply_structure, extract_structure_examples,
                                      find_classes, scheme_of_class)

TS_CLASS = """export class Store {
  private cache = {};

  constructor() {
    this.init();
  }

  getItem(key) {
    return this.cache[key];
  }

  addItem(key, value) {
    this.cache[key] = value;
  }

  _evict() {
    this.cache = {};
  }
}
"""

PY_CLASS = """class Store:
    def __init__(self):
        self.cache = {}

    def get_item(self, key):
        return self.cache[key]

    def add_item(self, key, value):
        self.cache[key] = value

    def _evict(self):
        self.cache = {}
"""


class TestStructure(unittest.TestCase):
    def test_find_ts(self):
        classes = find_classes(TS_CLASS, clike)
        self.assertEqual(len(classes), 1)
        self.assertEqual([m["name"] for m in classes[0]["methods"]],
                         ["constructor", "getItem", "addItem", "_evict"])

    def test_find_py(self):
        classes = find_classes(PY_CLASS, logiclines)
        self.assertEqual(len(classes), 1)
        self.assertEqual([m["name"] for m in classes[0]["methods"]],
                         ["__init__", "get_item", "add_item", "_evict"])

    def test_scheme_detection(self):
        cls = find_classes(TS_CLASS, clike)[0]
        self.assertEqual(scheme_of_class(cls, "clike"), "ctor_first")

    def test_apply_alpha_ts(self):
        out = apply_structure(TS_CLASS, clike, "alpha")
        names = [m["name"] for m in find_classes(out, clike)[0]["methods"]]
        self.assertEqual(names, ["constructor", "_evict", "addItem", "getItem"])

    def test_apply_alpha_py_compiles(self):
        out = apply_structure(PY_CLASS, logiclines, "alpha")
        compile(out, "<t>", "exec")
        names = [m["name"] for m in find_classes(out, logiclines)[0]["methods"]]
        self.assertEqual(names, ["__init__", "_evict", "add_item", "get_item"])

    def test_private_last(self):
        out = apply_structure(PY_CLASS, logiclines, "private_last")
        names = [m["name"] for m in find_classes(out, logiclines)[0]["methods"]]
        self.assertEqual(names[-1], "_evict")

    def test_extract_examples(self):
        ex = extract_structure_examples(TS_CLASS, clike)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["n_methods"], 4)


if __name__ == "__main__":
    unittest.main()
