import unittest

from codeprint.composer import Composer
from codeprint.syntax_gate import syntax_ok


class TestSyntaxGate(unittest.TestCase):
    def test_python(self):
        self.assertTrue(syntax_ok("x = 1\n", ".py"))
        self.assertFalse(syntax_ok("def (((:\n", ".py"))

    def test_ts_valid(self):
        self.assertTrue(syntax_ok("const x: number = 1;\n", ".ts"))
        self.assertTrue(syntax_ok('const c = <div className="a">{x}</div>;\n',
                                  ".tsx"))

    def test_ts_catches_trailing_comma_trap(self):
        # мина из аудита: trailing comma в if (...) — валидные токены,
        # битый синтаксис
        self.assertFalse(syntax_ok("if (a || b,) {\n  foo();\n}\n", ".ts"))

    def test_unknown_ext_passes(self):
        self.assertTrue(syntax_ok("любой текст", ".txt"))


class TestComposerGate(unittest.TestCase):
    def test_gate_rolls_back_broken_result(self):
        """Если трансформер сломал синтаксис — composer возвращает оригинал."""
        class BrokenNet:
            def encode(self, ex):
                return [1], [0.0] * 8
            def predict(self, ids, nums):
                return 1  # всегда «переносить»

        src = "if (alpha || beta) {\n  foo();\n}\n"
        # форсируем перенос if (...) — wrap_statement откажется (control flow),
        # но проверим сам гейт на искусственно битом трансформере:
        import codeprint.composer as comp
        orig_apply_wrapping = comp.apply_wrapping

        def bad_wrapping(source, backend, predict):
            return source.replace("if (alpha || beta) {",
                                  "if (alpha || beta,) {")

        comp.apply_wrapping = bad_wrapping
        try:
            c = Composer({"axes": {"wrapping": {"model": "x"}}})
            c._nets["x"] = BrokenNet()
            out = c.apply(src, ".ts")
            self.assertEqual(out, src)  # гейт откатил
        finally:
            comp.apply_wrapping = orig_apply_wrapping


if __name__ == "__main__":
    unittest.main()
