import unittest

from codeprint.planner import plan

TS_SRC = """import React from 'react';

export function processData(inputData) {
  const temporaryResult = inputData.map((item) => item.value * 2);
  const filteredData = temporaryResult.filter((x) => x > 10);
  return { data: filteredData };
}
"""


class TestPlanner(unittest.TestCase):
    def test_plan_returns_axes_subset(self):
        # профиль без нейронок и стилометра — только детерминированные оси
        import json
        import tempfile
        from pathlib import Path
        prof_dir = Path("profiles")
        test_prof = prof_dir / "_test_planner.json"
        test_prof.write_text(json.dumps({
            "name": "_test_planner",
            "axes": {
                "imports": {"scheme": "flat_alpha"},
                "idioms": {"ternary_max_chars": 50},
            },
        }), encoding="utf-8")
        try:
            result = plan("_test_planner", TS_SRC, ".ts")
            self.assertIn("chosen_axes", result)
            self.assertTrue(set(result["chosen_axes"]) <=
                            {"imports", "idioms"})
            self.assertGreaterEqual(len(result["history"]), 1)
        finally:
            test_prof.unlink()


if __name__ == "__main__":
    unittest.main()
