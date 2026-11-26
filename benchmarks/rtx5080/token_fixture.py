#!/usr/bin/env python3
"""Generate exact-length Inkling prompt IDs for both native benchmark CLIs."""

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

from jinja2 import Environment, nodes
from jinja2.ext import Extension
from tokenizers import Tokenizer

from qualify_long import SYSTEM, document


class GenerationBlock(Extension):
    tags = {"generation"}

    def parse(self, parser):
        line = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body).set_lineno(line)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixed_ids(tokenizer, template, occupied, task, effort, seed):
    marker = "__BENCHMARK_USER_ARCHIVE_1073266__"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": marker}]
    rendered = template.render(messages=messages, tools=[], reasoning_effort=effort,
                               add_generation_prompt=True, enable_thinking=effort != "none")
    prefix, suffix = rendered.split(marker)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    left, right = encode(prefix), encode(suffix)
    content_budget = occupied - len(left) - len(right)
    rows = max(1, (content_budget - 400) // 28)
    while rows:
        text, expected = document(rows, task, seed)
        content = encode(text)
        if len(content) <= content_budget:
            break
        rows -= max(1, (len(content) - content_budget) // 25)
    else:
        raise ValueError("Context is too short for the task")
    filler = encode("\nRoutine inventory handover recorded. No additional action required.\n")
    pad = content_budget - len(content)
    # Preserve all generated archive rows and the instruction by placing padding at their start.
    tokens = left + (filler * ((pad + len(filler) - 1) // len(filler)))[:pad] + content + right
    assert len(tokens) == occupied
    return tokens, rows, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--occupied", type=int, action="append", required=True)
    parser.add_argument("--output-tokens", type=int, default=384)
    parser.add_argument("--task", choices=("decode", "code", "recall"), default="decode")
    parser.add_argument("--effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"), default="none")
    parser.add_argument("--seed", type=int, default=5080)
    args = parser.parse_args()
    if min(args.occupied) < 1024 or args.output_tokens <= 0 or len(set(args.occupied)) != len(args.occupied):
        parser.error("Use unique occupied counts of at least 1,024 and a positive output budget")
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer_path = args.model_dir / "tokenizer.json"
    template_path = args.model_dir / "chat_template.jinja"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    environment = Environment(trim_blocks=True, lstrip_blocks=True, extensions=[GenerationBlock])
    template = environment.from_string(template_path.read_text(encoding="utf-8"))
    rows = []
    for count in args.occupied:
        ids, archive_rows, expected = fixed_ids(tokenizer, template, count, args.task, args.effort, args.seed)
        canonical = array("I", ids)
        if sys.byteorder != "little":
            canonical.byteswap()
        assert canonical.itemsize == 4
        tag = f"{args.task}-{count}"
        basalt = args.out / f"{tag}.ids"
        twinkling = args.out / f"{tag}.jsonl"
        basalt.write_text(",".join(map(str, ids)) + "\n", encoding="ascii")
        twinkling.write_text(json.dumps({"name": tag, "prompt": ids, "output": [0] * args.output_tokens}) + "\n")
        rows.append({"tag": tag, "occupied_tokens": count, "output_tokens": args.output_tokens,
                     "archive_rows": archive_rows, "canonical_ids_sha256": digest(canonical.tobytes()),
                     "expected": expected, "basalt": basalt.name, "twinkling": twinkling.name})
    manifest = {"seed": args.seed, "effort": args.effort, "task": args.task,
                "generator_sha256": digest(Path(__file__).read_bytes()),
                "tokenizer_sha256": digest(tokenizer_path.read_bytes()), "template_sha256": digest(template_path.read_bytes()),
                "cases": rows}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
