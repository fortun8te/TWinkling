#!/usr/bin/env python3
"""Run a fresh Pi editing task against an isolated loopback Inkling server on macOS."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

from qualify_long import CHECKS, check_code, document


FIXTURE = "def select_ticket(rows):\n    return rows[0]['id'] if rows else None\n"
PROMPT = """Read tickets.py and fix select_ticket using your file tools. Edit tickets.py itself.
Only status "open" qualifies. Choose the highest numeric priority, breaking ties by the
smallest lexicographic id. Return the selected id, or None if there is no match.
The function takes rows and must use one return expression, bracket field access, and
no imports or external helpers. Builtins min, max, sorted, any and all can be used.
Work entirely in this repository. Your answer can be brief once the edit is complete.
"""


def sandbox(root, node, pi_dir, port):
    writable = json.dumps(str(root))
    node_bin = json.dumps(str(node))
    install_paths = [node.parent.parent]
    cellar = next((parent for parent in node.parents if parent.name == "Cellar"), None)
    if cellar:
        install_paths.extend((cellar, cellar.parent / "opt"))
    read_paths = "".join(f" (subpath {json.dumps(str(path))})" for path in (
        root, *install_paths, pi_dir, "/System", "/usr", "/Library/Apple",
        "/private/etc", "/private/var/db/dyld",
    ))
    remote = json.dumps(f"localhost:{port}")
    return f"""(version 1)
(allow default)
(deny process-exec (require-not (literal {node_bin})))
(deny file-read-data (require-not (require-any {read_paths} (literal "/") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))))
(deny file-write* (require-not (require-any (subpath {writable}) (literal "/dev/null"))))
(deny network-outbound (require-not (remote tcp {remote})))
(deny network-inbound)
"""


def models(base_url, model, context, max_tokens):
    return {"providers": {"local-trial": {
        "baseUrl": base_url.rstrip("/"),
        "api": "openai-completions",
        "apiKey": "local-trial",
        "compat": {"supportsDeveloperRole": False, "maxTokensField": "max_tokens", "thinkingFormat": "openai"},
        "models": [{
            "id": model, "name": model, "contextWindow": context, "maxTokens": max_tokens,
            "input": ["text"], "reasoning": True,
            "thinkingLevelMap": {"minimal": None, "low": "low", "medium": "medium", "high": "high",
                                 "xhigh": None, "max": None},
            "cost": {key: 0 for key in ("input", "output", "cacheRead", "cacheWrite")},
        }],
    }}}


def trace_summary(path):
    counts = {"events": 0, "malformed_events": 0, "tool_calls": {}, "tool_errors": 0,
              "assistant_messages": 0, "max_occupied_tokens": None, "generated_tokens": 0}
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("expected object")
            except ValueError:
                counts["malformed_events"] += 1
                continue
            counts["events"] += 1
            if event.get("type") == "tool_execution_start":
                name = event.get("toolName") or "unknown"
                counts["tool_calls"][name] = counts["tool_calls"].get(name, 0) + 1
            if event.get("type") == "tool_execution_end" and event.get("isError"):
                counts["tool_errors"] += 1
            message = event.get("message") or {}
            if event.get("type") != "message_end" or message.get("role") != "assistant":
                continue
            counts["assistant_messages"] += 1
            usage = message.get("usage") or {}
            if "input" in usage:
                occupied = sum(usage.get(key, 0) or 0 for key in ("input", "cacheRead", "cacheWrite"))
                counts["max_occupied_tokens"] = max(counts["max_occupied_tokens"] or 0, occupied)
                counts["generated_tokens"] += usage.get("output") or 0
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", "--url", default="http://127.0.0.1:18367/v1")
    ap.add_argument("--model", default="twinkling-trial")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", choices=("off", "low", "medium", "high"), default="medium")
    ap.add_argument("--archive-rows", type=int, default=0,
                    help="insert a deterministic synthetic archive into the agent prompt")
    ap.add_argument("--seed", type=int, default=5080)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    url = urlsplit(args.base_url)
    if (url.scheme, url.hostname, url.path.rstrip("/")) != ("http", "127.0.0.1", "/v1") or (
            url.username or url.password or url.query or url.fragment or not url.port):
        ap.error("base-url must be http://127.0.0.1:PORT/v1")
    if not 0 < args.max_tokens < args.context or args.timeout <= 0 or args.archive_rows < 0:
        ap.error("invalid timeout, token budget, or archive size")
    cli = Path.home() / "tools/pi/node_modules/.bin/pi"
    node_found = shutil.which("node")
    seatbelt = Path("/usr/bin/sandbox-exec")
    if not cli.is_file() or not node_found or not seatbelt.is_file():
        ap.error("requires the local Pi installation, Node.js, and macOS sandbox-exec")
    node = Path(node_found).resolve()
    pi_dir = (Path.home() / "tools/pi").resolve()
    out = args.out.resolve()
    trace = out.with_suffix(".events.jsonl")
    errors = out.with_suffix(".stderr.txt")
    if len({trace, errors, out}) != 3 or any(path.exists() for path in (out, trace, errors)):
        ap.error("choose a fresh receipt path")
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT
    if args.archive_rows:
        archive, _ = document(args.archive_rows, "code", args.seed)
        archive = archive[:archive.rindex("OBSOLETE DRAFT: replace")]
        prompt = archive + "\n" + PROMPT

    with tempfile.TemporaryDirectory(prefix="twinkling-pi-") as directory:
        scratch = Path(directory).resolve()
        repo, profile, home, temp = (scratch / name for name in ("repo", "profile", "home", "tmp"))
        for path in (repo, profile, home, temp):
            path.mkdir()
        target = repo / "tickets.py"
        target.write_text(FIXTURE, encoding="utf-8")
        (profile / "models.json").write_text(json.dumps(models(args.base_url, args.model, args.context,
                                                              args.max_tokens)), encoding="utf-8")
        env = {"HOME": str(home), "TMPDIR": str(temp), "PI_CODING_AGENT_DIR": str(profile),
               "PATH": f"{node.parent}:/usr/bin:/bin", "LANG": "en_US.UTF-8", "PI_OFFLINE": "1", "OPENSSL_CONF": "/dev/null"}
        command = [str(seatbelt), "-p", sandbox(scratch, node, pi_dir, url.port),
                   str(node), str(cli.resolve()), "-p", "--mode", "json", "--no-session",
                   "--no-extensions", "--no-skills", "--no-context-files", "--no-mcp",
                   "--no-prompt-templates", "--tools", "read,edit,write",
                   "--provider", "local-trial", "--model", args.model, "--thinking", args.thinking]
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > 96000:
            prompt_input = {"input": prompt_bytes}
        else:
            command.append(prompt)
            prompt_input = {"stdin": subprocess.DEVNULL}
        start = time.monotonic()
        timed_out = False
        with trace.open("xb") as stdout, errors.open("xb") as stderr:
            try:
                status = subprocess.run(command, cwd=repo, env=env, stdout=stdout, stderr=stderr,
                                        timeout=args.timeout, **prompt_input).returncode
            except subprocess.TimeoutExpired:
                status, timed_out = None, True
        elapsed = time.monotonic() - start
        functional_pass = (target.is_file() and not target.is_symlink()
                           and target.stat().st_size <= 2000
                           and check_code(target.read_text(encoding="utf-8", errors="replace")))
    summary = trace_summary(trace)
    edited = bool(summary["tool_calls"].get("edit", 0) + summary["tool_calls"].get("write", 0))
    passed = bool(status == 0 and edited and functional_pass)
    receipt = {"endpoint": args.base_url, "model": args.model, "thinking": args.thinking,
               "context_window": args.context, "output_limit": args.max_tokens, "archive_rows": args.archive_rows,
               "seed": args.seed, "exit_code": status, "timed_out": timed_out,
               "wall_s": round(elapsed, 3), "functional_pass": bool(functional_pass),
               "functional_cases": len(CHECKS), "agent_completed": passed,
               "trace": str(trace), "stderr": str(errors), **summary}
    with out.open("x", encoding="utf-8") as file:
        json.dump(receipt, file, indent=2)
        file.write("\n")
    print(json.dumps(receipt))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
