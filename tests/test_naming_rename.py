import unittest

from codeprint import clike, logiclines
from codeprint.naming_axis import (make_renamer, rename_locals,
                                   rename_locals_python)

VOCAB = ["data", "payload", "entity", "meta", "result", "buffer", "handle"]


def renamer(seed="t"):
    return make_renamer(VOCAB, seed)


class TestRenamePython(unittest.TestCase):
    def test_local_renamed_and_compiles(self):
        src = ("def calc(input_data):\n"
               "    temp_result = input_data * 2\n"
               "    return temp_result\n")
        out = rename_locals_python(src, renamer())
        compile(out, "<t>", "exec")
        self.assertNotIn("temp_result", out)
        self.assertIn("input_data", out)  # параметры v1 не трогаем

    def test_strings_and_attrs_untouched(self):
        src = ('def f():\n'
               '    my_value = "my_value"\n'
               '    obj.my_value = 1\n'
               '    return my_value\n')
        out = rename_locals_python(src, renamer())
        compile(out, "<t>", "exec")
        self.assertIn('"my_value"', out)      # строка — контент
        self.assertIn("obj.my_value", out)    # атрибут — чужое

    def test_deterministic_with_seed(self):
        src = "def f():\n    some_var = 1\n    return some_var\n"
        self.assertEqual(rename_locals_python(src, make_renamer(VOCAB, "k")),
                         rename_locals_python(src, make_renamer(VOCAB, "k")))

    def test_shadowed_name_skipped(self):
        src = ("def outer():\n"
               "    shared = 1\n"
               "    def inner(shared):\n"
               "        return shared\n"
               "    return inner(shared)\n")
        out = rename_locals_python(src, renamer())
        compile(out, "<t>", "exec")
        self.assertIn("shared", out)  # перекрыто во вложенной — не трогаем

    def test_keyword_call_survives(self):
        src = ("def f():\n"
               "    opts_value = 1\n"
               "    return g(opts_value=opts_value)\n"
               "def g(opts_value=0):\n"
               "    return opts_value\n")
        out = rename_locals_python(src, renamer())
        compile(out, "<t>", "exec")
        # kwarg-ключ — контракт вызываемой функции, не меняется
        self.assertIn("opts_value=", out)


class TestRenameClike(unittest.TestCase):
    def test_simple_const_renamed(self):
        src = "const userData = fetch(id);\nconsole.log(userData);\n"
        out = rename_locals(src, clike, renamer())
        self.assertNotIn("userData", out)

    def test_shorthand_prop_skipped(self):
        src = ("const userData = 1;\n"
               "const payload = { userData };\n")
        out = rename_locals(src, clike, renamer())
        self.assertIn("userData", out)  # shorthand — опасно, пропускаем

    def test_exported_skipped(self):
        src = "const userData = 1;\nexport { userData };\n"
        out = rename_locals(src, clike, renamer())
        self.assertIn("userData", out)

    def test_template_literal_skipped(self):
        src = "const userData = 1;\nconst s = `${userData}`;\n"
        out = rename_locals(src, clike, renamer())
        self.assertIn("userData", out)

    def test_property_access_skipped(self):
        src = "const userData = { a: 1 };\nconst x = obj.userData;\nconsole.log(userData);\n"
        out = rename_locals(src, clike, renamer())
        # obj.userData — не тронуто при любом исходе
        self.assertIn("obj.userData", out)


if __name__ == "__main__":
    unittest.main()
