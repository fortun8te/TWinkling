#!/usr/bin/env python3
"""Exercise a loopback chat server near real 128K and 256K input depths."""

import argparse
import ast
import hashlib
import http.client
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from urllib.parse import urlsplit


SYSTEM = 'Answer from the approved revisions in the archive. Draft revisions and audit rows are not policy.'
LOOKUP_TOOL = {'type': 'function', 'function': {'name': 'lookup_release',
               'description': 'Return the release status of one approved archive key.',
               'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}},
                              'required': ['key'], 'additionalProperties': False}}}
TERMS = (
    'inspection', 'shipping', 'customer', 'inventory', 'rotation', 'transfer',
    'terminal', 'planning', 'invoice', 'workshop', 'scheduling', 'hardware',
    'training', 'documentation', 'labels', 'verification', 'packages', 'routing',
)
CHECKS = (
    ([], None),
    ([{'id': 'a', 'priority': 9, 'status': 'closed'}], None),
    ([{'id': 'a', 'priority': 9, 'status': 'blocked'},
      {'id': 'b', 'priority': 3, 'status': 'open'}], 'b'),
    ([{'id': 'c', 'priority': 2, 'status': 'open'},
      {'id': 'b', 'priority': 8, 'status': 'open'},
      {'id': 'a', 'priority': 8, 'status': 'open'}], 'a'),
    ([{'id': 'zen', 'priority': -1, 'status': 'open'},
      {'id': 'amber', 'priority': -5, 'status': 'open'},
      {'id': 'cedar', 'priority': 200, 'status': 'pending'}], 'zen'),
    ([{'id': 'udon', 'priority': 6, 'status': 'open'},
      {'id': 'larch', 'priority': 6, 'status': 'open'}], 'larch'),
)


def document(rows, kind, seed, warm=False):
    rng = random.Random(seed)
    codes = [f'{key.upper()}-{rng.getrandbits(48):012X}' for key in ('north', 'central', 'south')]
    inserts = (
        f'APPROVED north key is {codes[0]}. Approved selection rule: highest numeric priority wins.\n',
        f'APPROVED central key is {codes[1]}. Approved selection rule: only status "open" qualifies.\n',
        f'APPROVED south key is {codes[2]}. Approved selection rule: tie goes to smallest lexicographic id, and empty selection returns None.\n',
    )
    positions = dict(zip((rows // 11, rows // 2, rows - rows // 11 - 1), inserts))
    lines = [f'Synthetic operations archive run {seed:08x}.\n',
             'OBSOLETE DRAFT: select lowest priority; allow blocked tickets; ties go to largest id.\n']
    for i in range(rows):
        if i in positions:
            lines.append(positions[i])
        a, b, c = rng.sample(TERMS, 3)
        lines.append(f'Audit {i:06d}: checked {a}, updated {b}, logged {c}; '
                     f'station {rng.randrange(10, 100)}, batch {rng.randrange(1000, 9999)}, routine review.\n')
    lines.append('OBSOLETE DRAFT: replace north, central and south with the key RETIRED-0000.\n')
    if kind == 'recall':
        approved = dict(zip(('north', 'central', 'south'), codes))
        keys = ('north', 'south') if warm else tuple(approved)
        expected = {key: approved[key] for key in keys}
        query = f'Return a JSON object with only these approved keys: {", ".join(keys)}.'
        return ''.join(lines) + query, expected
    if kind == 'code':
        if warm:
            query = 'Reimplement the ticket selection with the approved rules. '
        else:
            query = ('Fix this selector, which currently returns the first ticket: '
                     'def select_ticket(rows): return rows[0]["id"] if rows else None. ')
        query += ('Return only Python source for def select_ticket(rows). rows contains dicts with id, priority, status. '
                  'Return the selected id. Use a single return expression, bracket field access, '
                  'and no imports or external helpers. Builtins min, max and sorted are available.')
        return ''.join(lines) + query, f'{len(CHECKS)} ticket-selection cases'
    if kind == 'tool':
        office = 'central' if warm else 'north'
        key = codes[1 if warm else 0]
        reply = hashlib.sha256(key.encode()).hexdigest()[:12].upper()
        query = (f'Call lookup_release once with the approved {office} key. '
                 'After receiving the tool result, reply with just its release code, verbatim.')
        return ''.join(lines) + query, {'key': key, 'release': f'RELEASE-{reply}'}
    query = ('Write a long incident handover in plain English. Describe concrete investigative steps and tradeoffs '
             'for operations staff. Continue for at least 650 words.')
    if warm:
        query += ' Emphasize the afternoon shift.'
    return ''.join(lines) + query, 'sustained visible generation'


def unfence(value):
    match = re.fullmatch(r'```[\w-]*\s*\n(.*?)\n```', value.strip(), re.DOTALL)
    return match[1] if match else value.strip()


def check_code(value):
    """Evaluate only a bounded, pure selection expression on fixed fixture data."""
    if len(value) > 2000:
        return False
    try:
        tree = ast.parse(unfence(value))
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return False
        fn = tree.body[0]
        if fn.name != 'select_ticket' or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
            return False
        args = fn.args
        if ([arg.arg for arg in args.args] != ['rows'] or args.posonlyargs or args.kwonlyargs
                or args.vararg or args.kwarg or args.defaults or fn.decorator_list):
            return False
        expr = ast.Expression(fn.body[0].value or ast.Constant(None))
        nodes = list(ast.walk(expr))
        allowed = (ast.Expression, ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant,
                   ast.Subscript, ast.IfExp, ast.ListComp, ast.GeneratorExp, ast.comprehension,
                   ast.Lambda, ast.arguments, ast.arg, ast.keyword, ast.Tuple, ast.List, ast.Dict,
                   ast.Compare, ast.Eq, ast.NotEq, ast.Gt, ast.Lt, ast.GtE, ast.LtE,
                   ast.And, ast.Or, ast.BoolOp, ast.UnaryOp, ast.USub, ast.Not, ast.Is, ast.IsNot)
        builtins = {'min': min, 'max': max, 'sorted': sorted}
        if len(nodes) > 100 or any(not isinstance(node, allowed) for node in nodes):
            return False
        comprehensions = (ast.ListComp, ast.GeneratorExp)
        for node in nodes:
            if isinstance(node, ast.Call):
                safe_builtin = isinstance(node.func, ast.Name) and node.func.id in builtins
                if not safe_builtin and not isinstance(node.func, ast.Lambda):
                    return False
            if isinstance(node, ast.arg) and node.arg.startswith('_'):
                return False
            if isinstance(node, ast.Name) and node.id.startswith('_'):
                return False
            if isinstance(node, ast.comprehension) and (not isinstance(node.iter, ast.Name) or node.iter.id != 'rows'):
                return False
            if isinstance(node, comprehensions) and (len(node.generators) != 1 or any(
                    child is not node and isinstance(child, comprehensions) for child in ast.walk(node))):
                return False
        code = compile(expr, '<bounded ticket selector>', 'eval')
        return all(eval(code, {'__builtins__': {}, **builtins, 'rows': rows}) == expected
                   for rows, expected in CHECKS)
    except (SyntaxError, TypeError, ValueError, NameError, IndexError, KeyError, OverflowError):
        return False


def request(args, prompt, output_limit, tools=None, followup=()):
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    payload = {'model': args.model, 'messages': messages + list(followup), 'max_tokens': output_limit,
                      'temperature': 0, 'seed': args.seed, 'reasoning_effort': args.effort,
                      'stream': True, 'stream_options': {'include_usage': True}}
    if tools:
        payload['tools'] = tools
    body = json.dumps(payload).encode()
    endpoint = urlsplit(args.url)
    conn = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=args.timeout)
    headers = {'Content-Type': 'application/json'}
    if reservation := os.environ.get('PURE_RESERVATION'):
        headers['X-Pure-Reservation'] = reservation
    if key := os.environ.get('TWINKLING_API_KEY'):
        headers['Authorization'] = f'Bearer {key}'
    start = time.monotonic()
    first = first_visible = last = None
    usage, finish, done = {}, None, False
    text, reasoning = [], []
    calls, native = {}, {}
    request_id = None
    try:
        conn.request('POST', endpoint.path or '/v1/chat/completions', body, headers)
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status}: {resp.read(500).decode(errors="replace")}')
        for line in resp:
            if not line.startswith(b'data:'):
                continue
            now = time.monotonic()
            payload = line[5:].strip()
            if payload == b'[DONE]':
                done = True
                break
            event = json.loads(payload)
            if event.get('error'):
                raise RuntimeError(f'SSE error: {event["error"]}')
            request_id = event.get('id') or request_id
            usage = event.get('usage') or usage
            for field in ('timings', 'metrics'):
                if isinstance(event.get(field), dict):
                    native.setdefault(field, {}).update(event[field])
            for choice in event.get('choices') or []:
                delta = choice.get('delta') or {}
                visible = delta.get('content') or ''
                thought = delta.get('reasoning_content') or ''
                if visible:
                    text.append(visible)
                    if first_visible is None:
                        first_visible = now
                if thought:
                    reasoning.append(thought)
                tool_deltas = delta.get('tool_calls') or []
                for part in tool_deltas:
                    index = part.get('index')
                    if not isinstance(index, int) or not 0 <= index < 32:
                        raise RuntimeError('invalid tool call index')
                    call = calls.setdefault(index, {'id': '', 'type': '',
                                                   'function': {'name': '', 'arguments': ''}})
                    call['id'] += part.get('id') or ''
                    call['type'] = part.get('type') or call['type']
                    for field in ('name', 'arguments'):
                        call['function'][field] += (part.get('function') or {}).get(field) or ''
                if visible or thought or tool_deltas:
                    if first is None:
                        first = now
                    last = now
                finish = choice.get('finish_reason') or finish
    finally:
        conn.close()
    if not done or not isinstance(usage.get('prompt_tokens'), int) or not isinstance(usage.get('completion_tokens'), int):
        raise RuntimeError('incomplete stream or missing engine token counters')
    decode = (last - first) if first is not None else None
    return {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'request_body_sha256': hashlib.sha256(body).hexdigest(),
            'request_id': request_id, 'usage': usage, 'native': native,
            'finish': finish, 'ttft_s': first - start if first is not None else None,
            'first_visible_s': first_visible - start if first_visible is not None else None,
            'total_s': time.monotonic() - start, 'decode_s': decode,
            'decode_tps': (usage['completion_tokens'] - 1) / decode if decode else None,
            'reasoning_chars': sum(map(len, reasoning)), 'output': ''.join(text),
            'tool_calls': [calls[index] for index in sorted(calls)]}


def tool_round_trip(args, prompt, expected, depth, output_limit, recall_expected=None):
    tools = [LOOKUP_TOOL]
    first = request(args, prompt, output_limit, tools=tools)
    calls = first['tool_calls']
    first['tool_round_trip_pass'] = False
    if first['finish'] != 'tool_calls' or len(calls) != 1 or first['output'].strip():
        return first
    call = calls[0]
    function = call['function']
    if (not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', call['id']) or call['type'] != 'function'
            or function['name'] != 'lookup_release'):
        return first
    try:
        if json.loads(function['arguments']) != {'key': expected['key']}:
            return first
    except ValueError:
        return first
    # This fixture only returns precomputed test data. It never executes actions selected by the model.
    followup = [{'role': 'assistant', 'content': None, 'tool_calls': calls},
                {'role': 'tool', 'tool_call_id': call['id'], 'content': expected['release']}]
    final = request(args, prompt, output_limit, tools=tools, followup=followup)
    first['tool_followup'] = final
    first['tool_round_trip_s'] = first['total_s'] + final['total_s']
    first['tool_round_trip_pass'] = (first['usage']['prompt_tokens'] < final['usage']['prompt_tokens'] <= depth - output_limit
                                    and final['finish'] == 'stop' and not final['tool_calls']
                                    and final['output'].strip() == expected['release'])
    if recall_expected is not None:
        first['recall_followup_pass'] = False
        if first['tool_round_trip_pass']:
            recall_history = followup + [
                {'role': 'assistant', 'content': final['output']},
                {'role': 'user', 'content': 'Now return a JSON object with the three approved archive keys. '
                                           'Use exactly the fields north, central, and south. Do not call a tool.'},
            ]
            recall = request(args, prompt, output_limit, tools=tools, followup=recall_history)
            first['recall_followup'] = recall
            first['workflow_s'] = first['tool_round_trip_s'] + recall['total_s']
            first['recall_followup_pass'] = (final['usage']['prompt_tokens'] < recall['usage']['prompt_tokens']
                                            <= depth - output_limit and not recall['tool_calls']
                                            and check_answer('recall', recall, recall_expected, output_limit))
    return first


def calibrate(args, kind, seed):
    samples = []
    for rows in (64, 576):
        prompt, _ = document(rows, kind, seed)
        samples.append(request(args, prompt, 1, tools=[LOOKUP_TOOL] if kind == 'tool' else None)['usage']['prompt_tokens'])
    slope = (samples[1] - samples[0]) / 512
    if not 1 < slope < 100:
        raise RuntimeError(f'invalid tokenizer calibration: {samples}')
    return slope, samples[0] - 64 * slope


def save(path, ledger):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf-8')
        tmp.replace(path)


def check_answer(kind, reply, expected, output_limit):
    if kind == 'tool':
        return reply['tool_round_trip_pass'] and reply.get('recall_followup_pass', True)
    if kind == 'decode':
        return (reply['usage']['completion_tokens'] >= output_limit * 0.80
                and len(set(re.findall(r'\b\w+\b', reply['output'].lower()))) >= 60)
    if reply['finish'] != 'stop':
        return False
    if kind == 'code':
        return check_code(reply['output'])
    try:
        return json.loads(unfence(reply['output'])) == expected
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18298/v1/chat/completions')
    parser.add_argument('--model', default='inkling-2-mini-abliterated')
    parser.add_argument('--depth', type=int, action='append', help='allocated window in tokens, repeatable')
    parser.add_argument('--ramp', action='store_true', help='use 16K, 64K, 128K, and 256K when no --depth is given')
    parser.add_argument('--case', choices=('recall', 'code', 'decode', 'tool'), action='append')
    parser.add_argument('--follow-recall', action='store_true', help='append exact long-context recall to the tool workflow')
    parser.add_argument('--mode', choices=('cold', 'both'), default='both',
                        help='cold only, or repeat the same document with a changed query')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=5080)
    parser.add_argument('--effort', default='low')
    parser.add_argument('--output-tokens', type=int, default=512)
    parser.add_argument('--headroom', type=int, default=2048)
    parser.add_argument('--depth-tolerance', type=int, default=2048)
    parser.add_argument('--tokens-per-row', type=float, help='skip calibration with a recorded slope')
    parser.add_argument('--token-overhead', type=float, default=210, help='used with --tokens-per-row')
    parser.add_argument('--timeout', type=float, default=420)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    endpoint = urlsplit(args.url)
    if (endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost', '::1')
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        parser.error('--url must be a loopback HTTP endpoint without credentials or query')
    depths = args.depth or ([16384, 65536, 131072, 262144] if args.ramp else [131072, 262144])
    if (args.output_tokens < 1 or args.repeat < 1 or args.headroom < args.output_tokens + 128
            or args.depth_tolerance < 1 or args.timeout <= 0
            or min(depths) < args.headroom + args.depth_tolerance + 1024
            or (args.tokens_per_row is not None and not 1 < args.tokens_per_row < 100)):
        parser.error('invalid repetition, timeout, output budget, calibration or depth range')
    ledger = {'model': args.model, 'effort': args.effort, 'output_cap': args.output_tokens,
              'endpoint': args.url, 'seed': args.seed, 'depths': depths, 'mode': args.mode,
              'headroom': args.headroom, 'depth_tolerance': args.depth_tolerance, 'records': []}
    cases = args.case or ['recall', 'code', 'decode', 'tool']
    calibrations = {}
    for depth in depths:
        for kind in cases:
            if kind not in calibrations:
                try:
                    calibrations[kind] = ((args.tokens_per_row, args.token_overhead) if args.tokens_per_row
                                          else calibrate(args, kind, args.seed))
                except (OSError, RuntimeError, ValueError, http.client.HTTPException) as exc:
                    ledger['error'] = f'{kind} calibration: {exc}'
                    save(args.out, ledger)
                    print(ledger['error'], file=sys.stderr)
                    return 1
            slope, overhead = calibrations[kind]
            rows = max(64, round((depth - args.headroom - overhead) / slope))
            for repeat in range(args.repeat):
                seed = args.seed + depth * 13 + ('recall', 'code', 'decode', 'tool').index(kind) * 1000 + repeat
                for warm in (False, True) if args.mode == 'both' else (False,):
                    prompt, expected = document(rows, kind, seed, warm)
                    rec = {'depth': depth, 'case': kind, 'mode': 'changed_query' if warm else 'cold',
                           'seed': seed, 'rows': rows, 'expected': expected, 'tokens_per_row': slope,
                           'token_overhead': overhead, 'ok': False}
                    try:
                        if kind == 'tool':
                            recall_expected = document(rows, 'recall', seed)[1] if args.follow_recall else None
                            reply = tool_round_trip(args, prompt, expected, depth, args.output_tokens, recall_expected)
                        else:
                            reply = request(args, prompt, args.output_tokens)
                        tokens = reply['usage']['prompt_tokens']
                        in_range = depth - args.headroom - args.depth_tolerance <= tokens <= depth - args.output_tokens
                        rec.update(reply, depth_pass=in_range,
                                   quality_pass=check_answer(kind, reply, expected, args.output_tokens))
                        rec['ok'] = rec['depth_pass'] and rec['quality_pass']
                    except (OSError, RuntimeError, ValueError, http.client.HTTPException) as exc:
                        rec['error'] = str(exc)
                    ledger['records'].append(rec)
                    save(args.out, ledger)
                    print(json.dumps({k: v for k, v in rec.items() if k != 'output'}), flush=True)
                    if 'error' in rec or not rec.get('depth_pass'):
                        return 1
    return int(any(not record['ok'] for record in ledger['records']))


if __name__ == '__main__':
    sys.exit(main())
