import unittest

from codeprint import clike, logiclines
from codeprint.naming_axis import extract_naming_examples, split_words


class TestNaming(unittest.TestCase):
    def test_split_words(self):
        self.assertEqual(split_words("fetchTemplateParts"),
                         ["fetch", "template", "parts"])
        self.assertEqual(split_words("fetch_template_parts"),
                         ["fetch", "template", "parts"])
        self.assertEqual(split_words("HTMLElement"), ["html", "element"])

    def test_python_skips_strings_comments_attrs(self):
        src = ('# комментарий project\n'
               'name = "строка server"\n'
               'obj.attr_value = 1\n'
               'myVariable = compute_value(my_param)\n'
               'second_value = 1\nthird_value = 2\nfourth_value = 3\n'
               'fifth_value = 4\nsixth_value = 5\nseventh_value = 6\n'
               'eighth_value = 7\n')
        ex = extract_naming_examples(src, logiclines)
        self.assertEqual(len(ex), 1)
        words = ex[0]["words"]
        self.assertIn("variable", words)      # myVariable
        self.assertIn("param", words)         # my_param
        self.assertNotIn("server", words)     # строковый литерал — не имя
        self.assertNotIn("project", words)    # комментарий — не имя
        self.assertNotIn("attr", words)       # obj.attr — чужое

    def test_ts_identifiers(self):
        src = ("const myData = fetchData(userId);\n"
               "const panelTitle = `заголовок ${userId}`;\n"
               "obj.someProp = 1;\n"
               "// комментарий hiddenword\n"
               "function computeTotal(itemsList) { return 1; }\n"
               "let anotherOne = 2;\nlet thirdValue = 3;\nlet fourthVal = 4;\n"
               "let fifthItem = 5;\nlet sixthThing = 6;\nlet seventhBit = 7;\n")
        ex = extract_naming_examples(src, clike)
        self.assertEqual(len(ex), 1)
        words = ex[0]["words"]
        self.assertIn("data", words)
        self.assertIn("total", words)
        self.assertNotIn("some", words)        # obj.someProp пропущен
        self.assertNotIn("hiddenword", words)  # комментарий

    def test_small_file_skipped(self):
        self.assertEqual(extract_naming_examples("x = 1\n", logiclines), [])


if __name__ == "__main__":
    unittest.main()
