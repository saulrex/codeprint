#!/usr/bin/env python3
"""app — веб-интерфейс codeprint поверх wizard/harvest.

Запуск: python app.py [--port 7100]  (или: npm run dev)

Страница проводит по шагам wizard'а: репозиторий -> выбор своих подписей ->
дедлайн -> сбор. Сохранёнки складываются в datasets/<имя-репо>/ рядом с app.py.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codeprint.harvest import main as harvest_main
from wizard import list_authors, run_git

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"

PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>codeprint — сбор датасета стиля</title>
<style>
  body { font-family: system-ui, sans-serif; background: #14151a; color: #e8e8ec;
         max-width: 760px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 20px; } h2 { font-size: 15px; color: #9aa; margin-top: 28px; }
  input[type=text] { width: 100%; padding: 8px; background: #1d1f26; color: inherit;
         border: 1px solid #333; border-radius: 6px; box-sizing: border-box; }
  button { padding: 8px 16px; background: #3b6eea; color: #fff; border: 0;
         border-radius: 6px; cursor: pointer; margin-top: 8px; }
  button:disabled { background: #444; cursor: wait; }
  label { display: block; padding: 3px 0; color: #c8ccd4; }
  pre { background: #0d0e12; padding: 12px; border-radius: 6px; overflow: auto;
        font-size: 12px; white-space: pre-wrap; }
  .card { background: #1d1f26; border-radius: 6px; padding: 10px 14px; margin: 6px 0; }
  .muted { color: #778; font-size: 12px; }
</style>
</head>
<body>
<h1>codeprint: сбор датасета стиля автора</h1>

<h2>1/4 Репозиторий</h2>
<input type="text" id="repo" placeholder="C:\\path\\to\\repo" value="">
<button onclick="scan()">Найти авторов</button>

<h2>2/4 Кто из них ты?</h2>
<div id="authors"><span class="muted">сначала укажи репозиторий</span></div>

<h2>3/4 Дедлайн (позже этой даты — «нейронка», не берём)</h2>
<input type="text" id="before" placeholder="YYYY-MM-DD">

<h2>4/4 Собрать</h2>
<button id="run" onclick="run()">Собрать датасет</button>
<pre id="log" style="display:none"></pre>

<h2>Сохранённые датасеты</h2>
<div id="datasets"></div>

<script>
async function scan() {
  const repo = document.getElementById('repo').value.trim();
  const r = await fetch('/api/authors?repo=' + encodeURIComponent(repo));
  const data = await r.json();
  const box = document.getElementById('authors');
  if (data.error) { box.innerHTML = '<span class="muted">' + data.error + '</span>'; return; }
  box.innerHTML = data.authors.map(a =>
    `<label><input type="checkbox" value="${a.email}"> ${a.count} коммитов — ${a.name} &lt;${a.email}&gt;</label>`
  ).join('');
}

async function run() {
  const repo = document.getElementById('repo').value;
  const emails = [...document.querySelectorAll('#authors input:checked')].map(c => c.value);
  const before = document.getElementById('before').value;
  const log = document.getElementById('log');
  const btn = document.getElementById('run');
  log.style.display = 'block'; log.textContent = 'работаю...\\n';
  btn_off(btn => btn.textContent = 'работаю...');
  function btn_off(f) { btn.disabled = true; f(btn); }
  try {
    const r = await fetch('/api/run', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({repo, emails, before}) });
    const data = await r.json();
    log.textContent += data.log + (data.error || '');
  } catch (e) { log.textContent += 'ошибка: ' + e; }
  btn.disabled = false; btn.textContent = 'Собрать датасет';
  loadDatasets();
}

async function loadDatasets() {
  const r = await fetch('/api/datasets');
  const data = await r.json();
  document.getElementById('datasets').innerHTML = data.datasets.map(d =>
    `<div class="card"><b>${d.name}</b> — ${d.examples} примеров, до ${d.before}
     <div class="muted">${d.dir}</div></div>`
  ).join('') || '<span class="muted">пока пусто</span>';
}
loadDatasets();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: str, ctype: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False),
                   "application/json; charset=utf-8")

    def log_message(self, *a):  # тишина в консоли
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif u.path == "/api/authors":
            repo = Path(parse_qs(u.query).get("repo", [""])[0].strip('"')).expanduser()
            try:
                run_git(repo, "rev-parse", "--git-dir")
                authors = [{"count": c, "name": n, "email": e}
                           for c, n, e in list_authors(repo)[:20]]
                self._json({"authors": authors})
            except Exception as e:
                self._json({"error": f"не git-репозиторий: {e}"}, 400)
        elif u.path == "/api/datasets":
            out = []
            if DATASETS.is_dir():
                for mf in sorted(DATASETS.glob("*/*/manifest.json")):
                    d = mf.parent
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    out.append({
                        "name": f"{d.parent.name} (до {m.get('before', '?')})",
                        "dir": str(d),
                        "before": m.get("before", "?"),
                        "examples": sum(m.get("total_examples", {}).values()),
                    })
            self._json({"datasets": out})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/run":
            self._json({"error": "not found"}, 404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            repo = Path(str(body["repo"]).strip('"')).expanduser()
            emails = [str(e) for e in body["emails"]]
            before = str(body["before"])
            dt.date.fromisoformat(before)
            if not emails:
                raise ValueError("не выбрано ни одной подписи")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self._json({"error": f"невалидный запрос: {e}"}, 400)
            return

        out_dir = DATASETS / repo.name / before  # каждый дедлайн — своя папка
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = ["--repo", str(repo), "--before", before, "--out", str(out_dir)]
        for e in emails:
            argv += ["--author", e]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = harvest_main(argv)
        if rc == 0:
            (out_dir / "wizard.json").write_text(json.dumps({
                "repo": str(repo), "emails": emails, "before": before,
                "ran_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "via": "web-ui",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        self._json({"rc": rc, "log": buf.getvalue(), "out": str(out_dir)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int,
                    default=int(__import__("os").environ.get("PORT", 7100)))
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"codeprint app: http://localhost:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
