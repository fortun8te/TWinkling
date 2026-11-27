"""Draw the long-context before/after charts as SVG from results.json (stdlib only)."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
r = json.loads((here / "results.json").read_text())
BG, GRID, INK, MUTED = "#141820", "#2b3542", "#edf5fc", "#8fa1b5"
COL = {"start": "#4a5666", "phase1": "#547ba6", "now": "#7fc4ff"}
NAMES = {"start": "Start", "phase1": "Phase 1", "now": "Now"}
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"


def pct(a, b):
    v = (b - a) / a * 100
    return f"{v:+.0f}%"


def grouped(title, sub, data, unit, better_low, fname):
    W, H, L, T, B = 760, 360, 64, 78, 300
    groups = list(data)
    top = max(max(v) for v in data.values()) * 1.12
    step = next(s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200) if s * 5 >= top)
    ticks = -(-int(top) // step)
    vmax = step * ticks
    gw = (W - L - 30) / len(groups)
    bw = gw / 5
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' {FONT}>",
           f"<rect width='{W}' height='{H}' rx='14' fill='{BG}'/>",
           f"<text x='28' y='36' fill='{INK}' font-size='19' font-weight='650'>{title}</text>",
           f"<text x='28' y='58' fill='{MUTED}' font-size='13'>{sub}</text>"]
    for i in range(ticks + 1):
        v = step * i
        y = B - (B - T) * i / ticks
        out.append(f"<line x1='{L}' x2='{W-24}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}'/>")
        out.append(f"<text x='{L-10}' y='{y+4:.1f}' fill='{MUTED}' font-size='11' text-anchor='end'>{v:.0f}</text>")
    for gi, g in enumerate(groups):
        x0 = L + gi * gw + bw
        for ci, c in enumerate(r["columns"]):
            v = data[g][ci]
            h = (B - T) * v / vmax
            x = x0 + ci * bw
            out.append(f"<rect x='{x+3:.1f}' y='{B-h:.1f}' width='{bw-6:.1f}' height='{h:.1f}' rx='4' fill='{COL[c]}'/>")
            out.append(f"<text x='{x+bw/2:.1f}' y='{B-h-6:.1f}' fill='{INK if c=='now' else MUTED}' font-size='11.5' text-anchor='middle'>{v:g}</text>")
        d = pct(data[g][0], data[g][-1])
        out.append(f"<text x='{x0+1.5*bw:.1f}' y='{B+22}' fill='{INK}' font-size='13' font-weight='600' text-anchor='middle'>{g}</text>")
        out.append(f"<text x='{x0+1.5*bw:.1f}' y='{B+40}' fill='#7fc4ff' font-size='12.5' font-weight='650' text-anchor='middle'>{d}</text>")
    lx = W - 24
    for c in reversed(r["columns"]):
        lx -= 78
        out.append(f"<rect x='{lx}' y='27' width='10' height='10' rx='2' fill='{COL[c]}'/><text x='{lx+15}' y='36' fill='{MUTED}' font-size='12'>{NAMES[c]}</text>")
    out.append("</svg>")
    (here / fname).write_text("\n".join(out))


grouped("Decode speed", "tok/s on a real coding task, higher is better", r["decode_tps"], "tok/s", False, "decode.svg")
grouped("Time to first token", "seconds, cold prefill of the full prompt, lower is better", r["first_token_s"], "s", True, "first-token.svg")

# Summary strip: start -> now with percent, one row per metric.
rows = [(f"Decode {k}", v[0], v[-1], "tok/s") for k, v in r["decode_tps"].items()]
rows += [(f"First token {k}", v[0], v[-1], "s") for k, v in r["first_token_s"].items()]
rows.append(("Load to ready", r["load_to_ready_s"][0], r["load_to_ready_s"][-1], "s"))
W, rh, top = 760, 34, 70
H = top + rh * len(rows) + 24
o = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' {FONT}>",
     f"<rect width='{W}' height='{H}' rx='14' fill='{BG}'/>",
     f"<text x='28' y='36' fill='{INK}' font-size='19' font-weight='650'>RTX 5080, start of tuning vs now</text>",
     f"<text x='28' y='56' fill='{MUTED}' font-size='13'>250 W, official weights, same fixtures. Bar length is the improvement factor.</text>"]
for i, (name, a, b, u) in enumerate(rows):
    y = top + i * rh
    factor = b / a if u == "tok/s" else a / b
    w = min(factor / 4.0, 1) * 300
    o.append(f"<text x='28' y='{y+21}' fill='{INK}' font-size='13.5'>{name}</text>")
    o.append(f"<text x='300' y='{y+21}' fill='{MUTED}' font-size='13' text-anchor='end'>{a:g} → <tspan fill='{INK}' font-weight='650'>{b:g}</tspan> {u}</text>")
    o.append(f"<rect x='320' y='{y+10}' width='300' height='12' rx='6' fill='{GRID}'/>")
    o.append(f"<rect x='320' y='{y+10}' width='{w:.1f}' height='12' rx='6' fill='#7fc4ff'/>")
    o.append(f"<text x='{W-28}' y='{y+21}' fill='#7fc4ff' font-size='13.5' font-weight='700' text-anchor='end'>{pct(a,b)} · {factor:.1f}×</text>")
o.append("</svg>")
(here / "summary.svg").write_text("\n".join(o))

# Where a 16K decode step goes.
prof = r["decode_profile_16k_pct"]
W, H = 760, 150
o = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' {FONT}>",
     f"<rect width='{W}' height='{H}' rx='14' fill='{BG}'/>",
     f"<text x='28' y='36' fill='{INK}' font-size='19' font-weight='650'>Where a 16K decode round goes</text>",
     f"<text x='28' y='56' fill='{MUTED}' font-size='13'>Nsight timeline, PDL-corrected. ~73% of the 893 GB/s byte ceiling.</text>"]
shades = ["#7fc4ff", "#5c9fd6", "#4a7fae", "#3d6488", "#344c66", "#2d3d50"]
x, tot, span = 28.0, sum(prof.values()), W - 56
for i, (k, v) in enumerate(prof.items()):
    w = span * v / tot
    o.append(f"<rect x='{x:.1f}' y='72' width='{w-2:.1f}' height='22' rx='4' fill='{shades[i]}'/>")
    lx = 28 + (i % 3) * 240
    ly = 118 + (i // 3) * 20
    o.append(f"<rect x='{lx}' y='{ly-9}' width='9' height='9' rx='2' fill='{shades[i]}'/><text x='{lx+15}' y='{ly}' fill='{MUTED}' font-size='12.5'>{k} <tspan fill='{INK}' font-weight='600'>{v:g}%</tspan></text>")
    x += w
o[1] = o[1].replace(f"height='{H}'", f"height='{H+8}'")
o[0] = o[0].replace(f"0 0 {W} {H}", f"0 0 {W} {H+8}").replace(f"height='{H}'", f"height='{H+8}'")
o.append("</svg>")
(here / "decode-profile.svg").write_text("\n".join(o))
print("wrote", *sorted(p.name for p in here.glob("*.svg")))
