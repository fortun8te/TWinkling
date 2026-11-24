#!/usr/bin/env python3
"""End-to-end check of image input through a running ingine server (OpenAI API).

Sends the fixtures of native/tests/golden/vision/fixtures as base64 data: URIs and checks the answers and
the reuse of the prompt caches: the colours of solid images (two in one message, in order), the text of a
rendered line, a follow-up turn (the image is not read again), another image at the same place, streaming
with usage, rejected requests, and, with --long, an image at the end of a long text prompt.

Standard library only. Exit status 0 when every check passes.

  python3 tools/vision_smoke.py --base-url http://127.0.0.1:8003/v1 [--long 200000]
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "native" / "tests" / "golden" / "vision" / "fixtures"
FAILURES = 0


def check(ok, what):
    global FAILURES
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    FAILURES += not ok


def data_uri(name):
    kind = {"png": "png", "jpg": "jpeg", "gif": "gif", "webp": "webp"}[name.rsplit(".", 1)[1]]
    return f"data:image/{kind};base64," + base64.b64encode((FIXTURES / name).read_bytes()).decode()


def image(name):
    return {"type": "image_url", "image_url": {"url": data_uri(name)}}


def text(value):
    return {"type": "text", "text": value}


class Server:
    def __init__(self, base_url, model):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def post(self, body, timeout=600):
        request = urllib.request.Request(self.base_url + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def chat(self, messages, max_tokens=64, **extra):
        """Returns (status, answer text in lower case or the error message, usage, seconds)."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "enable_thinking": False, **extra}
        start = time.time()
        status, raw = self.post(body)
        seconds = time.time() - start
        reply = json.loads(raw)
        if status != 200:
            return status, reply["error"]["message"], None, seconds
        return status, (reply["choices"][0]["message"]["content"] or "").lower(), reply["usage"], seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--long", type=int, default=0, metavar="TOKENS",
                        help="also put an image after about this many tokens of text (e.g. 200000)")
    args = parser.parse_args()

    with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=10) as response:
        card = json.loads(response.read())["data"][0]
    print(f"model {card['id']}, context {card.get('max_model_len')}, input {card.get('input_modalities')}")
    check("image" in card.get("input_modalities", []), "the model card lists image input")
    server = Server(args.base_url, card["id"])
    colour = "What colour is this image? Answer with one word."

    status, answer, usage, _ = server.chat([{"role": "user", "content": [image("solid_red_40x40.png"), text(colour)]}])
    check(status == 200 and "red" in answer, f"a red image is red: {answer!r}")
    red_prompt = usage["prompt_tokens"] if usage else 0
    status, answer, usage, _ = server.chat([{"role": "user", "content": [image("solid_blue_40x40.png"), text(colour)]}])
    check(status == 200 and "blue" in answer, f"a blue image at the same place is blue: {answer!r}")
    check(usage and usage["prompt_tokens"] == red_prompt, "two one-token images cost the same")

    both = "Name the colour of the first image, then the colour of the second image. Two words."
    status, answer, _, _ = server.chat([{"role": "user", "content": [image("solid_red_40x40.png"),
                                                                     image("solid_blue_40x40.png"), text(both)]}])
    check(status == 200 and "red" in answer and "blue" in answer and answer.index("red") < answer.index("blue"),
          f"red then blue: {answer!r}")
    status, answer, _, _ = server.chat([{"role": "user", "content": [image("solid_blue_40x40.png"),
                                                                     image("solid_red_40x40.png"), text(both)]}])
    check(status == 200 and "red" in answer and "blue" in answer and answer.index("blue") < answer.index("red"),
          f"blue then red: {answer!r}")

    status, answer, usage, _ = server.chat([{"role": "user", "content": [
        image("text_inkling42_320x80.png"), text("What is written in this image? Reply with the text only.")]}])
    check(status == 200 and "inkling" in answer and "42" in answer, f"reads the text of an image: {answer!r}")

    for name in ("photo_160x120.jpg", "palette_80x40.gif", "rgba_64x64.png", "lossy_160x120.webp",
                 "lossless_alpha_72x56.webp", "animated_80x80.webp"):
        status, answer, _, _ = server.chat([{"role": "user", "content": [image(name), text("Describe this image briefly.")]}],
                                           max_tokens=24)
        check(status == 200 and len(answer) > 0, f"{name}: {answer[:60]!r}")

    # a large image, then a follow-up turn: the second request reads only what is new
    describe = [{"role": "user", "content": [image("big_1720x1720.png"), text("Describe its colours in one sentence.")]}]
    status, first, usage, t_first = server.chat(describe, max_tokens=48)
    check(status == 200 and usage["prompt_tokens"] > 1849, f"an 1849-token image ({t_first:.2f} s): {first[:80]!r}")
    follow = describe + [{"role": "assistant", "content": first},
                         {"role": "user", "content": "Is it a photograph? Answer yes or no."}]
    status, answer, _, t_follow = server.chat(follow, max_tokens=8)
    check(status == 200 and ("no" in answer or "yes" in answer), f"a follow-up turn ({t_follow:.2f} s): {answer!r}")
    check(t_follow < max(0.5 * t_first, 0.15), f"the follow-up did not read the image again ({t_follow:.2f} s vs {t_first:.2f} s)")

    # streaming with usage
    body = {"model": server.model, "messages": [{"role": "user", "content": [image("solid_red_40x40.png"), text(colour)]}],
            "max_tokens": 16, "temperature": 0, "enable_thinking": False, "stream": True,
            "stream_options": {"include_usage": True}}
    status, raw = server.post(body)
    events = [json.loads(line[6:]) for line in raw.decode().split("\n") if line.startswith("data: {")]
    streamed = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e.get("choices"))
    usages = [e["usage"] for e in events if e.get("usage")]
    check(status == 200 and "red" in streamed.lower() and usages and usages[-1]["prompt_tokens"] == red_prompt,
          f"streaming: {streamed!r}, usage {usages[-1] if usages else None}")

    # rejected requests
    for content, needle in (([{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}], "data: uris"),
                            ([{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}], "png, jpeg, webp"),
                            ([text("a <|unused_200054|> b"), image("solid_red_40x40.png")], "placeholder")):
        status, message, _, _ = server.chat([{"role": "user", "content": content}])
        check(status == 400 and needle in message.lower(), f"rejected with 400: {message[:90]!r}")

    if args.long:
        # about 18 tokens per line; the passphrase is near the start, the image at the end
        lines = [f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(int(args.long / 18.0))]
        lines.insert(len(lines) // 50, "The passphrase is MARIGOLD-42.")
        question = "Two questions. What is the passphrase from the text? What is written in the image?"
        status, answer, usage, seconds = server.chat([{"role": "user", "content": [
            text("\n".join(lines)), image("text_inkling42_320x80.png"), text(question)]}], max_tokens=48)
        check(status == 200 and "marigold-42" in answer and "inkling" in answer,
              f"an image after {usage['prompt_tokens'] if usage else '?'} tokens ({seconds:.1f} s): {answer[:100]!r}")
        status, answer, usage, seconds = server.chat([{"role": "user", "content": [
            text("\n".join(lines[: len(lines) // 2])), image("big_1720x1720.png"), image("medium_720x680.png"),
            text("Describe the colours of the two images in one sentence.")]}], max_tokens=48)
        check(status == 200 and len(answer) > 10,
              f"two large images after a long text ({usage['prompt_tokens'] if usage else '?'} tokens, {seconds:.1f} s): {answer[:100]!r}")

    print("ok" if not FAILURES else f"{FAILURES} check(s) failed")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
