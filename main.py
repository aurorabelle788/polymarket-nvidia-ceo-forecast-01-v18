"""Polymarket NVIDIA CEO forecaster — v18.

5-way ensemble, weights [0.28, 0.28, 0.18, 0.06, 0.20]:
  leg 1 (0.28): EWMA-sigma fan, per-price-level half-life (8/10/15/22 min),
                cross-market log-log level-fit shrink, sum correction toward
                total 1.0, spike reversion, thin-95% tail widening.
  leg 2 (0.28): logit-space ensemble: 0.7 robust-logit forecaster
                (sigma from last-1440/60-min logit moves, left-skewed
                quantiles, up-move give-back median) + 0.3 gated
                price-space replica (sum-reversion gated to 0.8-1.2).
  leg 3 (0.18): pooled empirical h-step change table over the last 18h,
                vol-adaptive spread, SUM drift, net-jump median correction,
                short-window reversion.
  leg 4 (0.06): Gaussian reshape of the spike-reverted leg-1 rows.
  leg 5 (0.20): regime-adaptive sigma fan (EWMA 26-min half-life,
                level-fit shrink, regime calming, sum correction, per-market
                fatness), spike-reverted. No Laplace, no knot table.

Python 3.13, standard library only. Offline, deterministic, no network.
Reads $MAKEFUN_INPUT_DIR (else /input) and writes $MAKEFUN_OUTPUT_DIR
(else /output); one output file per input case.
"""

import json
import math
import os

MULT0 = (-1.8130133458944, -0.9521483940509615, -0.27,
         0.3914563862643396, 1.8623702217199494)
ZGAUSS = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
GRID = (1, 2, 3, 5, 7, 11, 16, 24, 36, 54, 81, 121, 180, 240)
W18 = (0.28, 0.28, 0.18, 0.06, 0.20)

# --- friend's regime-fan constants (leg 5) ---
FAN_MULT = (-2.007, -0.946, -0.403, 0.275, 1.854)
FAN_BAND = ((30, 1.2399), (120, 1.1139), (10 ** 9, 0.9215))
FAN_TABLE_W = (0.10, 0.10, 0.05, 0.35, 0.45)


def _median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _clean(row):
    row = [min(1.0, max(0.0, v)) for v in row]
    for j in range(1, 5):
        if row[j] < row[j - 1]:
            row[j] = row[j - 1]
    return [round(v, 6) for v in row]


def _clean_rows(rows):
    return [_clean(list(r)) for r in rows]


def _std(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return math.sqrt(sum((v - m) ** 2 for v in vals) / n)


def _q_at(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    pos = tau * (n - 1)
    lo = int(pos)
    hi = lo + 1 if lo + 1 < n else n - 1
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ------------------------------------------------------------------ spike
def _spike_shifts(prices, thresh=0.03, frac=1.0, tau=90.0):
    """Per-horizon reversion of the last 1-min move, if it was a spike.
    Returns None when the last move was not a spike."""
    d = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
    if abs(d) < thresh:
        return None
    return [-frac * d * (1.0 - math.exp(-(h + 1) / tau)) for h in range(240)]


def _apply_spike(rows_by_mid, case):
    out = {}
    for m in case["markets"]:
        mid = m["id"]
        rows = rows_by_mid[mid]
        shifts = _spike_shifts(m["prices"])
        if shifts is None:
            out[mid] = rows
            continue
        nr = []
        for h in range(240):
            corr = shifts[h]
            if corr == 0.0:
                nr.append(rows[h])
                continue
            nr.append(_clean([v + corr for v in rows[h]]))
        out[mid] = nr
    return out


# ------------------------------------------------------------------ leg 1
def _v10_rows(case, hl):
    """EWMA-sigma fan with level-fit shrink and sum correction (v10 core),
    one half-life for the whole case. PV multipliers (0.9 thin / 1.15 mid /
    0.8 favourite) are applied per market by price level."""
    markets = case["markets"]
    raw = []
    for m in markets:
        prices = m["prices"]
        rets = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        r = rets[-120:]
        if not r:
            raw.append(0.0025)
            continue
        d = 0.5 ** (1.0 / hl)
        ws = [d ** (len(r) - 1 - i) for i in range(len(r))]
        raw.append(math.sqrt(sum(w * v * v for w, v in zip(ws, r)) / sum(ws)))
    med = _median(raw)
    lx = [math.log(max(1e-6, m["prices"][-1] * (1.0 - m["prices"][-1])))
          for m in markets]
    ly = [math.log(max(1e-9, r)) for r in raw]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    sxx = sum((a - mx) ** 2 for a in lx)
    b1 = sxy / sxx if sxx > 0 else 0.0
    b0 = my - b1 * mx
    sigs = []
    for i in range(n):
        lv = math.exp(b0 + b1 * lx[i])
        t = lv * (1.0 - 0.4) + med * 0.4
        s = raw[i] * (1.0 - 0.6) + t * 0.6
        sigs.append(min(0.005, max(0.0002, s)) * 0.87)
    plain = [min(0.005, max(0.0002, r * (1.0 - 0.4) + med * 0.4)) * 0.87
             for r in raw]
    s_now = sum(m["prices"][-1] for m in markets)
    gap = (1.0 - s_now) * 1.15
    tot = sum(plain) or 1.0
    shifts = [gap * s / tot for s in plain]
    widths = [(h ** 0.47) * (1.08 if h <= 30 else 1.05 if h <= 120 else 0.95)
              for h in range(1, 241)]
    ramp = [min(1.0, h / 240.0) ** 0.7 for h in range(1, 241)]
    out = {}
    for idx, m in enumerate(markets):
        x = m["prices"][-1]
        sig = sigs[idx]
        if x < 0.10:
            sig *= 0.9
        elif x < 0.30:
            sig *= 1.15
        else:
            sig *= 0.8
        sh = shifts[idx]
        rows = []
        for hi in range(240):
            w = sig * widths[hi]
            c = x + sh * ramp[hi]
            row = [c + MULT0[j] * w for j in range(5)]
            rows.append([0.0 if v < 0.0 else 1.0 if v > 1.0 else round(v, 6)
                         for v in row])
        out[m["id"]] = rows
    return out


def _leg1(case):
    """Per-price-level half-life: very thin (<0.05) 8, thin (0.05-0.10) 10,
    mid (0.10-0.30) 15, favourite (>0.30) 22 minutes."""
    outs = {}
    for hl in (8.0, 10.0, 15.0, 22.0):
        outs[hl] = _v10_rows(case, hl)
    out = {}
    for m in case["markets"]:
        x = m["prices"][-1]
        if x < 0.05:
            hl = 8.0
        elif x < 0.10:
            hl = 10.0
        elif x < 0.30:
            hl = 15.0
        else:
            hl = 22.0
        out[m["id"]] = outs[hl][m["id"]]
    return out


def _widen95(rows_by_mid, case):
    """Widen the 95% quantile x1.5 for very thin markets (last < 0.05)."""
    out = {}
    for m in case["markets"]:
        mid = m["id"]
        rows = rows_by_mid[mid]
        if m["prices"][-1] >= 0.05:
            out[mid] = _clean_rows(rows)
            continue
        nr = []
        for r in rows:
            row = [r[0], r[1], r[2], r[3], r[2] + (r[4] - r[2]) * 1.5]
            nr.append(_clean(row))
        out[mid] = nr
    return out


# ------------------------------------------------------------------ leg 2
def _logit(p):
    p = min(0.9995, max(0.0005, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _leg2_sigma_a(prices):
    lg = [_logit(p) for p in prices]
    moves = [lg[i + 1] - lg[i] for i in range(len(lg) - 1)]

    def window_sigma(mv):
        if len(mv) < 10:
            return 0.0025
        absmv = [abs(d) for d in mv]
        mean_abs = sum(absmv) / len(absmv)
        s_mean = mean_abs / 0.7978845608028654
        med = _median(absmv)
        if med > 0:
            s_med = med / 0.6744897501960817
            return min(s_mean, s_med)
        return s_mean

    s1440 = window_sigma(moves[-1439:] if len(moves) >= 1439 else moves)
    s60 = window_sigma(moves[-59:] if len(moves) >= 59 else moves)
    sig = 0.5 * s1440 + 0.5 * s60
    return min(0.10, max(0.002, sig))


def _leg2_rows_a(prices):
    x = prices[-1]
    ref = prices[-61] if len(prices) >= 61 else prices[0]
    r = x - ref
    sig = _leg2_sigma_a(prices)
    big = x >= 0.10
    z = [-2.0, -1.0, 0.0, 0.4, 1.30] if big else [-1.80, -0.85, 0.0, 0.5, 1.50]
    rows = []
    for h in range(1, 241):
        if r > 0:
            med = x - (1.0 - math.exp(-h / 120.0)) * r
        else:
            med = x
        center = _logit(med)
        spread_mid = sig * (h ** 0.40)
        spread_tail = sig * (h ** 0.45) * 1.30
        row = []
        for j, zj in enumerate(z):
            sp = spread_tail if j in (0, 4) else spread_mid
            row.append(_sigmoid(center + zj * sp))
        rows.append(row)
    return rows


def _leg2_sigma_b(prices):
    moves = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    mv = moves[-720:]
    if len(mv) == 0:
        return 0.0002
    clipped = [min(0.05, max(-0.05, d)) for d in mv]
    mean_abs = sum(abs(d) for d in clipped) / len(clipped)
    sig = 0.75 * mean_abs
    return min(0.01, max(0.0002, sig))


def _leg2_rows_b(prices, s_sum):
    x = prices[-1]
    sig = _leg2_sigma_b(prices)
    z = [-2.0, -0.5, 0.0, 0.5, 2.0]
    gated = 0.8 <= s_sum <= 1.2
    target = x / s_sum if gated and s_sum != 0 else x
    rows = []
    for h in range(1, 241):
        if gated:
            med = x + (1.0 - math.exp(-h / 120.0)) * (target - x)
        else:
            med = x
        sp = sig * math.sqrt(h)
        row = [min(1.0, max(0.0, med + zj * sp)) for zj in z]
        rows.append(row)
    return rows


def _leg2(case):
    s_sum = sum(m["prices"][-1] for m in case["markets"])
    out = {}
    for m in case["markets"]:
        prices = m["prices"]
        a = _leg2_rows_a(prices)
        b = _leg2_rows_b(prices, s_sum)
        rows = []
        for h in range(240):
            row = [round(0.7 * a[h][j] + 0.3 * b[h][j], 6) for j in range(5)]
            rows.append(_clean(row))
        out[m["id"]] = rows
    return out


# ------------------------------------------------------------------ leg 3
def _leg3(case, window=1080, trim=0.02):
    table = {}
    for h in GRID:
        changes = []
        for m in case["markets"]:
            hist = m["prices"][-window:]
            if len(hist) - h < 20:
                continue
            changes.extend(hist[i + h] - hist[i] for i in range(len(hist) - h))
        if changes:
            changes.sort()
            table[h] = [_q_at(changes, t) for t in TAUS]

    def interp(h):
        keys = sorted(table)
        if not keys:
            return None
        if h <= keys[0]:
            return list(table[keys[0]])
        if h >= keys[-1]:
            return list(table[keys[-1]])
        lh = math.log(h)
        for a, b in zip(keys, keys[1:]):
            if a <= h <= b:
                la, lb = math.log(a), math.log(b)
                w = (lh - la) / (lb - la) if lb > la else 0.0
                qa, qb = table[a], table[b]
                return [qa[j] * (1.0 - w) + qb[j] * w for j in range(5)]
        return list(table[keys[-1]])

    pooled_d1 = []
    for m in case["markets"]:
        hist = m["prices"][-window:]
        pooled_d1.extend(hist[i + 1] - hist[i] for i in range(len(hist) - 1))
    pooled_d1 = [max(-trim, min(trim, d)) for d in pooled_d1]
    pvol = _std(pooled_d1) if pooled_d1 else 0.0
    tail_inner = math.sqrt(0.85)
    ssum = sum(m["prices"][-1] for m in case["markets"])
    out = {}
    for m in case["markets"]:
        last = m["prices"][-1]
        hist = m["prices"]
        gap = (last / ssum - last) if ssum > 0 else 0.0
        recent = hist[-121:] if len(hist) >= 121 else hist
        mv = _std([recent[i + 1] - recent[i] for i in range(len(recent) - 1)])
        vf = 1.0
        if pvol > 0 and mv > 0:
            vf = max(0.4, min(2.0, (mv / pvol) ** 0.65))
        rows = []
        for h in range(1, 241):
            q = interp(h)
            if q is None:
                q = [0.0, 0.0, 0.0, 0.0, 0.0]
            drift = gap * (h / 240.0)
            row = [
                last + q[0] * 0.90 * vf + drift,
                last + q[1] * tail_inner * vf + drift,
                last + drift,
                last + q[3] * tail_inner * vf + drift,
                last + q[4] * 0.80 * vf + drift,
            ]
            rows.append(_clean(row))
        # net-jump median correction: signed sum of |1-min move| >= 0.05
        # over the last 60 minutes.
        tot = 0.0
        n = len(hist)
        for j in range(max(1, n - 60), n):
            d = hist[j] - hist[j - 1]
            if abs(d) >= 0.05:
                tot += d
        if tot != 0.0:
            for h in range(1, 241):
                if tot > 0:
                    corr = -0.5 * tot * (1.0 - math.exp(-h / 15.0))
                else:
                    corr = 0.3 * tot * (1.0 - math.exp(-h / 30.0))
                if corr == 0.0:
                    continue
                row = rows[h - 1]
                rows[h - 1] = _clean([v + corr for v in row])
        # short-window reversion: only when the last price sits above its
        # 30-minute mean.
        w = hist[-30:] if len(hist) >= 30 else hist
        mean30 = sum(w) / len(w)
        dev = hist[-1] - mean30
        if dev > 0.0:
            for h in range(1, 241):
                corr = -0.5 * dev * (1.0 - math.exp(-h / 60.0))
                row = rows[h - 1]
                rows[h - 1] = _clean([v + corr for v in row])
        out[m["id"]] = rows
    return out


# ------------------------------------------------------------------ leg 4
def _leg4(spa_rows_by_mid):
    wspan = MULT0[4] - MULT0[0]
    out = {}
    for mid, rows in spa_rows_by_mid.items():
        nr = []
        for row in rows:
            w = (row[4] - row[0]) / wspan
            c = row[2] - MULT0[2] * w
            g = [c + ZGAUSS[j] * w * (1.30 if j in (0, 4) else 1.0)
                 for j in range(5)]
            nr.append(_clean(g))
        out[mid] = nr
    return out


# ------------------------------------------------------------------ leg 5
def _fan_band(h):
    for lim, f in FAN_BAND:
        if h <= lim:
            return f
    return 0.9215


def _leg5(case):
    """Regime-adaptive sigma fan: EWMA 26-min half-life, cross-market
    log-log level-fit shrink, regime calming (exponent -1.8), sum
    correction (strength 0.84, phased (h/240)^0.46), per-market fatness.
    No Laplace widening, no knot table."""
    stats = []
    for m in case["markets"]:
        prices = m["prices"]
        x = prices[-1]
        rets = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        r = rets[-120:] if len(rets) > 120 else rets
        if r:
            d = 0.5 ** (1.0 / 26.0)
            ws = [d ** (len(r) - 1 - i) for i in range(len(r))]
            raw = math.sqrt(sum(w * v * v for w, v in zip(ws, r)) / sum(ws))
            meanabs = sum(abs(v) for v in r) / len(r)
            act = sum(1 for v in r if v != 0.0)
        else:
            raw, meanabs, act = 0.0025, 0.0, 0
        stats.append({"id": m["id"], "x": x, "raw": raw,
                      "meanabs": meanabs, "act": act})
    raws = [s["raw"] for s in stats]
    med = _median(raws)
    xs = [s["x"] for s in stats]
    lx = [math.log(max(1e-6, x * (1.0 - x))) for x in xs]
    ly = [math.log(max(1e-9, r)) for r in raws]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    b1 = sxy / sxx if sxx else 0.0
    b0 = my - b1 * mx
    # regime factors (guard the case median with a 1e-9 floor)
    big_r = max(med, 1e-9)
    ref = 0.0003
    wf = min(1.6, max(0.6, (big_r / ref) ** -1.8))
    sf = min(3.0, max(0.1, (big_r / ref) ** -1.0))
    ratios = []
    for s in stats:
        if s["meanabs"] > 0:
            ratios.append(s["raw"] / s["meanabs"])
    med_ratio = _median(ratios) if ratios else 1.0
    if med_ratio <= 0:
        med_ratio = 1.0
    wts = [s["raw"] * (float(s["act"]) ** 0.65) for s in stats]
    tot_w = sum(wts) or 1.0
    s_total = sum(xs)
    gap = 1.0 - s_total
    out = {}
    for s, w in zip(stats, wts):
        x = s["x"]
        lv = math.exp(b0 + b1 * math.log(max(1e-6, x * (1.0 - x))))
        sig = 0.4 * s["raw"] + 0.6 * (0.6 * lv + 0.4 * med)
        sig = min(0.005, max(0.0001, sig)) * 0.9
        if s["meanabs"] > 0:
            fat = min(1.6, max(0.8, ((s["raw"] / s["meanabs"]) / med_ratio)
                                ** 0.6))
        else:
            fat = 1.0
        shift_share = gap * (w / tot_w) * 0.84 * sf
        rows = []
        for h in range(1, 241):
            ramp = (h / 240.0) ** 0.46
            c = x + shift_share * ramp
            hw = (h ** 0.46) * _fan_band(h) * wf
            row = []
            for j in range(5):
                off = FAN_MULT[j] * sig * hw
                if j in (0, 4):
                    off *= fat
                row.append(c + off)
            rows.append(row)
        out[s["id"]] = _clean_rows(rows)
    return out


# ------------------------------------------------------------------ ensemble
def _ensemble(leg_list, weights):
    out = {}
    for mid in leg_list[0]:
        rows = []
        for h in range(240):
            row = [round(sum(w * f[mid][h][j]
                             for w, f in zip(weights, leg_list)), 6)
                   for j in range(5)]
            rows.append(_clean(row))
        out[mid] = rows
    return out


def forecast_case(case):
    """Forecast one case: {market id: 240 rows x 5 quantiles}."""
    spa = _widen95(_apply_spike(_leg1(case), case), case)
    f9 = _leg2(case)
    f6w = _leg3(case)
    ga = _leg4(spa)
    fan = _apply_spike(_leg5(case), case)
    return _ensemble([spa, f9, f6w, ga, fan], W18)


# ------------------------------------------------------------------ I/O
def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    ins = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not ins:
        raise SystemExit("no .in file in " + in_dir)
    os.makedirs(out_dir, exist_ok=True)
    for fname in ins:
        with open(os.path.join(in_dir, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        if case.get("quantiles") != [0.05, 0.25, 0.5, 0.75, 0.95]:
            raise SystemExit("unexpected quantiles in " + fname)
        fc = forecast_case(case)
        cid = case.get("case") or os.path.splitext(fname)[0]
        # Format self-check before writing: every market exactly once,
        # exactly 240 rows of 5 finite numbers in [0, 1], non-decreasing.
        for m in case["markets"]:
            rows = fc[m["id"]]
            if len(rows) != 240:
                raise SystemExit("bad row count for " + m["id"])
            for row in rows:
                if len(row) != 5:
                    raise SystemExit("bad quantile count for " + m["id"])
                for j, v in enumerate(row):
                    if not (0.0 <= v <= 1.0) or v != v:
                        raise SystemExit("bad quantile value for " + m["id"])
                    if j and row[j] < row[j - 1]:
                        raise SystemExit("decreasing quantiles for " + m["id"])
        forecasts = [{"id": m["id"], "q": fc[m["id"]]}
                     for m in case["markets"]]
        out_path = os.path.join(out_dir, cid + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"forecasts": forecasts}, f, sort_keys=True,
                      separators=(",", ":"))
        if os.path.getsize(out_path) > 1048576:
            raise SystemExit("output exceeds 1,048,576 bytes: " + out_path)


if __name__ == "__main__":
    main()
