import unittest

from codeprint import clike, logiclines
from codeprint.imports_axis import (SCHEMES, apply_scheme, find_import_blocks,
                                    scheme_of, extract_imports_examples)

TS_SRC = """import React from 'react';
import { api } from './api';
import { Button } from './ui/Button';
import axios from 'axios';

import { helper } from './helpers';
import lodash from 'lodash';

export const x = 1;
"""


class TestImportBlocks(unittest.TestCase):
    def test_find_blocks_ts(self):
        blocks = find_import_blocks(TS_SRC, "clike")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]["lines"]), 6)
        self.assertEqual(blocks[0]["group_breaks"], 1)

    def test_scheme_detection(self):
        blocks = find_import_blocks(TS_SRC, "clike")
        self.assertEqual(scheme_of(blocks[0], "clike"), "grouped_asis")

    def test_scheme_flat_alpha(self):
        src = "import a from 'a';\nimport b from 'b';\nimport c from 'c';\nx = 1\n"
        blocks = find_import_blocks(src, "clike")
        self.assertEqual(scheme_of(blocks[0], "clike"), "flat_alpha")

    def test_apply_grouped_alpha(self):
        src = ("import { helper } from './helpers';\n"
               "import axios from 'axios';\n"
               "import { api } from './api';\n"
               "\n"
               "const x = 1;\n")
        out = apply_scheme(src, "grouped_alpha", "clike")
        lines = out.splitlines()
        # внешний (axios) первым, потом пустая, потом локальные по алфавиту
        self.assertEqual(lines[0], "import axios from 'axios';")
        self.assertEqual(lines[1], "")
        self.assertEqual(lines[2], "import { api } from './api';")
        self.assertEqual(lines[3], "import { helper } from './helpers';")

    def test_apply_flat_asis_keeps_order(self):
        src = ("import z from 'z';\nimport a from 'a';\nimport m from 'm';\n"
               "\nconst x = 1;\n")
        out = apply_scheme(src, "flat_asis", "clike")
        self.assertEqual(out.splitlines()[:3],
                         ["import z from 'z';", "import a from 'a';",
                          "import m from 'm';"])

    def test_python_blocks(self):
        src = ("import os\nimport sys\n\nimport numpy\nimport requests\n"
               "\nx = 1\n")
        blocks = find_import_blocks(src, "python")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(scheme_of(blocks[0], "python"), "grouped_alpha")

    def test_short_runs_ignored(self):
        src = "import os\nimport sys\n\nx = 1\n"
        self.assertEqual(find_import_blocks(src, "python"), [])

    def test_extract_examples(self):
        ex = extract_imports_examples(TS_SRC, clike)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["label"], SCHEMES.index("grouped_asis"))
        self.assertEqual(ex[0]["n_imports"], 6)

    def test_apply_is_idempotent(self):
        once = apply_scheme(TS_SRC, "grouped_alpha", "clike")
        self.assertEqual(once, apply_scheme(once, "grouped_alpha", "clike"))


if __name__ == "__main__":
    unittest.main()
