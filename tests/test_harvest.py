import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from codeprint.harvest import extract_blankline_examples, main as harvest_main

GIT = shutil.which("git")


class TestExtractBlanklines(unittest.TestCase):
    def test_labels_012(self):
        src = (
            "x = 1\n"
            "y = 2\n"          # граница 1: 0 пустых
            "\n"
            "z = 3\n"          # граница 2: 1 пустая
            "\n\n\n"
            "w = 4\n"          # граница 3: 3 пустые -> label 2 (кап)
        )
        ex = extract_blankline_examples(src)
        self.assertEqual([e["label"] for e in ex], [0, 1, 2])
        self.assertEqual(ex[2]["next_line"], "w = 4")

    def test_blank_inside_string_not_counted(self):
        src = 's = """a\n\n\nb"""\nx = 1\n'
        ex = extract_blankline_examples(src)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["label"], 0)
        self.assertEqual(ex[0]["next_line"], "x = 1")

    def test_features_present(self):
        ex = extract_blankline_examples("def f():\n    a = 1\n\n    return a\n")
        self.assertEqual(len(ex), 2)
        self.assertEqual(ex[1]["next_kw"], "return")
        self.assertEqual(ex[1]["label"], 1)
        self.assertTrue(ex[0]["prev_ends_colon"])

    def test_broken_file_yields_nothing(self):
        self.assertEqual(extract_blankline_examples("def (((:\n"), [])


def _git(repo: Path, *args: str, env: dict | None = None) -> None:
    full_env = {**os.environ, **(env or {})}
    subprocess.run([GIT, *args], cwd=repo, check=True,
                   capture_output=True, env=full_env)


@unittest.skipUnless(GIT, "git не найден")
class TestHarvestE2E(unittest.TestCase):
    """Фикстурный репо: мой старый коммит, чужой коммит, мой коммит ПОСЛЕ
    дедлайна. В датасет должен попасть только первый."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "fixture"
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.name", "t")
        _git(self.repo, "config", "user.email", "t@t")

        me = {"GIT_AUTHOR_NAME": "me", "GIT_AUTHOR_EMAIL": "me@x"}
        other = {"GIT_AUTHOR_NAME": "other", "GIT_AUTHOR_EMAIL": "o@x"}

        # c1: я, до дедлайна
        (self.repo / "a.py").write_text("x = 1\n\ny = 2\n", encoding="utf-8")
        _git(self.repo, "add", ".")
        _git(self.repo, "commit", "-q", "-m", "c1", env={
            **me, "GIT_AUTHOR_DATE": "2024-01-01T00:00:00",
            "GIT_COMMITTER_DATE": "2024-01-01T00:00:00"})

        # c2: чужой автор
        (self.repo / "b.py").write_text("z = 3\n", encoding="utf-8")
        _git(self.repo, "add", ".")
        _git(self.repo, "commit", "-q", "-m", "c2", env={
            **other, "GIT_AUTHOR_DATE": "2024-02-01T00:00:00",
            "GIT_COMMITTER_DATE": "2024-02-01T00:00:00"})

        # c3: я, но ПОСЛЕ дедлайна (файл изменён — другой blob)
        (self.repo / "a.py").write_text("x = 1\n\n\ny = 2\n", encoding="utf-8")
        _git(self.repo, "add", ".")
        _git(self.repo, "commit", "-q", "-m", "c3", env={
            **me, "GIT_AUTHOR_DATE": "2027-01-01T00:00:00",
            "GIT_COMMITTER_DATE": "2027-01-01T00:00:00"})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_filters_author_and_deadline(self):
        out = self.tmp / "out"
        rc = harvest_main([
            "--repo", str(self.repo),
            "--author", "me@x",
            "--before", "2026-01-01",
            "--out", str(out),
        ])
        self.assertEqual(rc, 0)

        examples = [json.loads(l) for l in
                    (out / "blanklines.jsonl").read_text(encoding="utf-8").splitlines()]
        # только a.py версии c1: одна граница, автор поставил 1 пустую строку
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["label"], 1)
        self.assertEqual(examples[0]["next_line"], "y = 2")

        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["repos"][0]["commits"], 1)
        self.assertEqual(manifest["repos"][0]["files"], 1)
        self.assertEqual(manifest["before"], "2026-01-01")

    def test_bad_date_rejected_early(self):
        rc = harvest_main([
            "--repo", str(self.repo), "--author", "me@x",
            "--before", "koga-to", "--out", str(self.tmp / "out2"),
        ])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
