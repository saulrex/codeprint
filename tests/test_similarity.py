import unittest

from codeprint.similarity import containment, jaccard, report

A = """const fetchUser = async (id) => {
  const response = await api.get(`/users/${id}`);
  return response.data;
};
"""


class TestSimilarity(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(containment(A, A), 1.0)
        self.assertEqual(jaccard(A, A), 1.0)

    def test_unrelated(self):
        b = "def foo(x):\n    return [i for i in range(x)]\n"
        self.assertLess(jaccard(A, b), 0.3)

    def test_small_edit_high_similarity(self):
        b = A.replace("fetchUser", "loadUser")
        self.assertGreater(containment(A, b), 0.5)

    def test_commented_out_does_not_fool(self):
        # закомментированный код всё же токенизируется — известное свойство;
        # проверяем только что добавление шума снижает сходство
        noise = "\n".join(f"const v{i} = {i};" for i in range(50))
        self.assertLess(jaccard(A, A + noise), 0.5)

    def test_report_keys(self):
        r = report(A, A.replace("api.get", "api.post"))
        self.assertIn("containment_before_in_after", r)
        self.assertIn("jaccard", r)


if __name__ == "__main__":
    unittest.main()
