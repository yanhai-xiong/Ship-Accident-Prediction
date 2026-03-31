"""Parse accident type and engineer tabular features from Chinese investigation reports."""

from __future__ import annotations

import re
from typing import Any

# Longer phrases first so e.g. 搁浅触礁 matches before 搁浅
_ACCIDENT_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("搁浅触礁", "搁浅触礁"),
    ("触碰", "碰撞"),
    ("碰撞", "碰撞"),
    ("搁浅", "搁浅"),
    ("触礁", "触礁"),
    ("火灾", "火灾"),
    ("爆炸", "爆炸"),
    ("沉没", "沉没"),
    ("溢油", "溢油"),
    ("浪损", "浪损"),
    ("风灾", "风灾"),
    ("机损", "机损"),
    ("机电设备损坏", "机损"),
    ("走锚", "走锚"),
    ("触损", "触损"),
    ("倾覆", "倾覆"),
    ("失踪", "失踪"),
    ("自沉", "沉没"),
]

# Line breaks often appear between "事故" and "调查报告", or between "事" and "故"
_TITLE_RE_JOIN = re.compile(r"轮\s*(.{1,60}?)\s*事故\s*\n*\s*调查报告", re.DOTALL)
_TITLE_RE_SPLIT = re.compile(r"轮\s*(.{1,60}?)\s*事\s*\n*\s*故\s*\n*\s*调查报告", re.DOTALL)
# First explicit accident phrase in header / 简介
_HEAD_ACCIDENT_RE = re.compile(
    r"(搁浅触礁|搁浅|碰撞|触礁|触碰|火灾|爆炸|沉没|溢油|浪损|风灾|机损|走锚|触损|倾覆|失踪)事故"
)


def _collapse_cjk_spaces(s: str) -> str:
    """Some PDFs insert spaces between every CJK glyph; tighten for regex matching."""
    t = re.sub(r"([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", r"\1", s)
    return re.sub(r"\s+", " ", t)

# Keywords for environment / operations / vessel (presence = 1)
_ENV_KEYS = [
    "大风",
    "浓雾",
    "能见度",
    "能见度不良",
    "横风",
    "横流",
    "急流",
    "潮汐",
    "潮流",
    "台风",
    "雷雨",
    "浪高",
]
_OPS_KEYS = ["靠泊", "离泊", "锚泊", "狭水道", "通航密集", "追越", "横越", "掉头"]
_VESSEL_KEYS = [
    "干货船",
    "油船",
    "油轮",
    "集装箱",
    "散货船",
    "客船",
    "拖轮",
    "起重船",
    "工程船",
    "液化气",
    "化学品船",
]
_HUMAN_KEYS = ["瞭望", "疲劳驾驶", "配员", "证书", "不适任", "指挥不当", "操作不当"]
# Narrative consequence wording (for keyword hits in pre-outcome features)
_CONSEQ_KEYS = ["人员伤亡", "死亡", "失踪", "水域污染", "海洋污染", "溢油"]

_AREA_KEYS = ["近海", "沿海", "内河", "港口", "锚地", "航道", "大洋"]
_EQUIPMENT_KEYS = ["AIS", "VDR", "雷达", "电子海图", "GPS", "ECDIS", "测深仪"]
_CARGO_KEYS = ["煤炭", "原油", "成品油", "集装箱", "散货", "液化", "件杂货", "矿石"]
_REGULATORY_KEYS = ["安全管理", "船舶检验", "适航证书", "配员证书", "最低安全配员"]
_PILOT_TUG_KEYS = ["引航", "引航员", "拖轮", "协助靠泊"]

# Classification societies / SOLAS (keyword hits, from report text only)
_CLASS_SOCIETY_KEYS = [
    "中国船级社",
    "CCS",
    "DNV",
    "ABS",
    "BV",
    "英国劳氏",
    "LR",
    "NK",
    "船级社",
]


def _count_keywords(text: str, keys: list[str]) -> int:
    return sum(1 for k in keys if k in text)


# Official-style tier phrases (longer first). Used to infer severity label from report text.
_SEVERITY_PHRASES: list[tuple[str, str]] = [
    ("特别重大", "特别重大"),
    ("较大事故", "较大"),
    ("较大等级", "较大"),
    ("重大事故", "重大"),
    ("重大等级", "重大"),
    ("一般等级", "一般"),
    ("一般事故", "一般"),
    ("小事故", "小事故"),
    ("较大", "较大"),
    ("重大", "重大"),
]


def infer_accident_severity(text: str) -> tuple[str | None, str | None]:
    """
    Infer regulatory accident tier from wording in the report (e.g. 一般等级、小事故).
    Returns (canonical_label, matched_span_or_None).
    """
    head = text[:30_000]
    head_norm = _collapse_cjk_spaces(head)
    for blob in (head_norm, head):
        for phrase, label in _SEVERITY_PHRASES:
            if phrase in blob:
                return label, phrase
    return None, None


_RE_LOSS_WAN = re.compile(r"直接经济损失[^0-9]{0,20}([0-9.]+)\s*万")


def _parse_int_patterns(text: str, patterns: list[str]) -> int:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return 0


def engineer_post_accident_features(text: str) -> dict[str, Any]:
    """
    Outcome-related fields extracted from the report text (loss, casualties, etc.).

    Used to **quantify** severity labels via ``severity_tier_from_outcomes`` and kept in the CSV
    for audit — **not** used as inputs when training severity or type (see YAML ``drop_always``).
    """
    t = text or ""
    head = t[:120_000]
    out: dict[str, Any] = {}

    lm = _RE_LOSS_WAN.search(head)
    if lm:
        try:
            out["num_direct_loss_wan"] = float(lm.group(1).replace(",", ""))
        except ValueError:
            out["num_direct_loss_wan"] = 0.0
    else:
        out["num_direct_loss_wan"] = 0.0

    out["num_deaths_reported"] = float(
        _parse_int_patterns(
            head,
            [r"(\d+)\s*人死亡", r"死亡\s*(\d+)\s*人", r"遇难\s*(\d+)\s*人"],
        )
    )
    out["num_missing_reported"] = float(
        _parse_int_patterns(
            head,
            [r"(\d+)\s*人失踪", r"失踪\s*(\d+)\s*人"],
        )
    )

    inj = ("受伤" in t) or ("轻伤" in t) or ("重伤" in t)
    out["flag_injury_mentioned"] = 1.0 if inj else 0.0
    out["post_outcome_keyword_hits"] = float(
        _count_keywords(
            t,
            ["直接经济损失", "人员伤亡", "死亡", "失踪", "水域污染", "海洋污染", "溢油", "沉没"],
        )
    )
    return out


def severity_tier_from_outcomes(post: dict[str, Any]) -> str | None:
    """
    Map extracted outcome fields to a regulatory-style tier label (**for training targets only**).

    Uses approximate bands inspired by 《水上交通事故统计办法》 and related MOT maritime loss
    announcements (casualty counts; direct loss in **万元**). Tune thresholds if you need
    stricter legal alignment.

    Returns ``None`` when there is no usable numeric outcome signal (all zeros) — callers may
    then fall back to text parsing.
    """
    loss = float(post.get("num_direct_loss_wan") or 0.0)
    deaths = int(float(post.get("num_deaths_reported") or 0))
    missing = int(float(post.get("num_missing_reported") or 0))
    inj = float(post.get("flag_injury_mentioned") or 0.0) >= 1.0
    lives = deaths + missing

    if lives >= 30:
        return "特别重大"
    if lives >= 10:
        return "重大"
    if lives >= 3:
        return "较大"
    if lives >= 1:
        return "一般"

    if inj:
        return "一般"

    if loss <= 0:
        return None

    # Direct economic loss (万元) — illustrative sea-transport bands (亿元 scale)
    if loss >= 30_000:
        return "特别重大"
    if loss >= 10_000:
        return "重大"
    if loss >= 2_000:
        return "较大"
    if loss >= 100:
        return "一般"
    return "小事故"


def resolve_accident_severity_training_label(
    text: str,
    post: dict[str, Any],
) -> tuple[str | None, str, str | None]:
    """
    Severity **label** for supervised training: prefer outcome-based tier, else phrase in report.

    Post-outcome columns in ``post`` quantify severity but must **not** be used as model inputs
    when predicting severity — only as the source of this label (or document extraction fallback).
    """
    tier = severity_tier_from_outcomes(post)
    if tier is not None:
        return tier, "outcomes", None
    sev, phrase = infer_accident_severity(text)
    if sev:
        return sev, "text", phrase
    return None, "", None


def infer_accident_type(text: str) -> tuple[str | None, str | None]:
    """
    Returns (canonical_type, raw_span) from title/header.
    If unknown, (None, raw or None).
    """
    head = text[:25_000]
    head_norm = _collapse_cjk_spaces(head)
    raw: str | None = None
    m = _TITLE_RE_JOIN.search(head_norm) or _TITLE_RE_SPLIT.search(head_norm)
    if not m:
        m = _TITLE_RE_JOIN.search(head) or _TITLE_RE_SPLIT.search(head)
    if m:
        raw = m.group(1).strip()
    span = raw or ""
    if not span:
        mh = _HEAD_ACCIDENT_RE.search(head)
        if mh:
            k = mh.group(1)
            if k == "触碰":
                return "碰撞", "触碰"
            for needle, canonical in _ACCIDENT_TYPE_KEYWORDS:
                if needle == k or needle in k:
                    return canonical, k
    if not span:
        return None, raw
    for needle, canonical in _ACCIDENT_TYPE_KEYWORDS:
        if needle in span:
            return canonical, raw
    for needle, canonical in _ACCIDENT_TYPE_KEYWORDS:
        if needle in head[:4000]:
            return canonical, raw
    return None, raw


_NUM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gross_ton", re.compile(r"总吨[：:]\s*([0-9,]+)")),
    ("net_ton", re.compile(r"净吨[：:]\s*([0-9,]+)")),
    ("main_engine_kw", re.compile(r"主机功率[：:]\s*([0-9,.]+)\s*[Kk]?[Ww]?")),
    ("length_m", re.compile(r"船长[：:]\s*([0-9.]+)\s*米")),
    (
        "draft_m",
        re.compile(r"(?:艏吃水|艉吃水|满载吃水|吃水)[：:]\s*([0-9.]+)\s*米"),
    ),
    ("speed_kn", re.compile(r"航速[：:]\s*([0-9.]+)\s*节")),
    ("visibility_m", re.compile(r"能见度[：:]\s*([0-9.]+)\s*米")),
    ("crew_onboard", re.compile(r"(?:在船人员|实际配员|配员)[：:]\s*([0-9]+)\s*人")),
]

_RE_WIND_RANGE = re.compile(r"(\d+)\s*[-~至]\s*(\d+)\s*级")
_RE_WIND_SINGLE = re.compile(r"(?:风|浪)[^。\n]{0,40}?(\d+)\s*级")


def _parse_first_float(pat: re.Pattern[str], text: str) -> float | None:
    m = pat.search(text)
    if not m:
        return None
    s = m.group(1).replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_wind_level_max(text: str, *, limit: int = 15_000) -> float | None:
    """First plausible wind force (Beaufort-style level) in the early report / weather lines."""
    s = text[:limit]
    m = _RE_WIND_RANGE.search(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return float(max(a, b))
    m = _RE_WIND_SINGLE.search(s)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 12:
            return float(v)
    return None


def _parse_accident_hour(text: str, *, limit: int = 12_000) -> float | None:
    """Hour 0–23 from common time-of-accident patterns near the start of the report."""
    s = text[:limit]
    m = re.search(r"(?:日| |\n)\s*(\d{4})\s*时", s)
    if m and m.group(1).isdigit() and len(m.group(1)) == 4:
        h = int(m.group(1)[:2])
        if 0 <= h <= 23:
            return float(h)
    m = re.search(r"(\d{1,2})\s*时\s*左右", s)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return float(h)
    return None


def _parse_year_built(text: str) -> float | None:
    head = text[:80_000]
    for pat in [
        r"建造完工日期[：:\s][^\n]{0,40}((?:19|20)\d{2})年",
        r"建成日期[：:\s]*((?:19|20)\d{2})",
        r"建造日期[：:\s][^\n]{0,40}((?:19|20)\d{2})年",
    ]:
        m = re.search(pat, head)
        if m:
            y = int(m.group(1))
            if 1950 <= y <= 2030:
                return float(y)
    return None


def _parse_accident_year_month(text: str) -> tuple[float | None, float | None]:
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text[:15_000])
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1990 <= y <= 2030 and 1 <= mo <= 12:
            return float(y), float(mo)
    return None, None


def _parse_beam_m(text: str) -> float | None:
    m = re.search(r"船宽[：:\s]*([0-9.]+)\s*米", text[:50_000])
    return float(m.group(1)) if m else None


def _parse_dwt(text: str) -> float | None:
    head = text[:50_000]
    for pat in [
        r"载重吨[：:\s]*([0-9,]+)",
        r"参考载货量[：:\s（(]*吨[）):：\s]*([0-9,]+)",
        r"载重[：:\s]*([0-9,]+)\s*吨",
    ]:
        m = re.search(pat, head)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _parse_lat_lon_decimal(text: str) -> tuple[float | None, float | None]:
    """First lat/lon pair in dd°mm.mm′ N/E form (common in CN reports)."""
    head = text[:25_000]
    m = re.search(
        r"(\d{1,2})°\s*(\d{1,2}\.?\d*)\s*[′'′]\s*([NS])[，,\s/、]{0,3}(\d{2,3})°\s*(\d{1,2}\.?\d*)\s*[′'′]\s*([EW])",
        head,
        re.I,
    )
    if not m:
        return None, None
    lat_d, lat_m, ns, lon_d, lon_m, ew = (
        float(m.group(1)),
        float(m.group(2)),
        m.group(3).upper(),
        float(m.group(4)),
        float(m.group(5)),
        m.group(6).upper(),
    )
    lat = lat_d + lat_m / 60.0
    if ns == "S":
        lat = -lat
    lon = lon_d + lon_m / 60.0
    if ew == "W":
        lon = -lon
    return lat, lon


def _parse_env_scalar(text: str, label: str, pattern: re.Pattern[str]) -> float | None:
    m = pattern.search(text[:45_000])
    if not m:
        return None
    raw = m.group(1).strip()
    if "-" in raw and raw.count("-") == 1 and not raw.startswith("-"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _engineer_hint_based_features(text: str, base: dict[str, Any]) -> dict[str, Any]:
    """
    Extra features aligned with vessel / environment / operational hints (v_*, E_*, O_*).
    Values are only set when regex finds a match in the report — no imputation beyond 0.0.
    """
    t = text or ""
    out: dict[str, Any] = {}

    yb = _parse_year_built(t)
    ay, am = _parse_accident_year_month(t)
    out["num_year_built"] = float(yb) if yb is not None else 0.0
    out["num_accident_year"] = float(ay) if ay is not None else 0.0
    out["num_accident_month"] = float(am) if am is not None else 0.0
    if yb is not None and ay is not None:
        age = max(0.0, min(80.0, float(ay - yb)))
        out["num_vessel_age_years"] = age
    else:
        out["num_vessel_age_years"] = 0.0

    bm = _parse_beam_m(t)
    dwt = _parse_dwt(t)
    out["num_beam_m"] = float(bm) if bm is not None else 0.0
    out["num_dwt_approx"] = float(dwt) if dwt is not None else 0.0

    lg = base.get("num_length_m") or 0.0
    gw = base.get("num_gross_ton") or 0.0
    kw = base.get("num_main_engine_kw") or 0.0
    df = base.get("num_draft_m") or 0.0

    out["ratio_length_to_beam"] = float(lg / bm) if bm and bm > 0 and lg > 0 else 0.0
    out["ratio_kw_per_gross_ton"] = float(kw / gw) if gw and gw > 0 and kw > 0 else 0.0
    out["ratio_draft_to_depth_hint"] = float(df / lg) if lg and lg > 0 and df > 0 else 0.0

    lat, lon = _parse_lat_lon_decimal(t)
    out["num_latitude"] = float(lat) if lat is not None else 0.0
    out["num_longitude"] = float(lon) if lon is not None else 0.0

    ws = _parse_env_scalar(
        t, "wind", re.compile(r"风速[：:\s]*([0-9.]+)\s*(?:米/秒|m/s)")
    )
    if ws is None:
        ws = _parse_env_scalar(t, "wind2", re.compile(r"(\d+\.?\d*)\s*米\s*/\s*秒"))
    out["num_wind_speed_ms"] = float(ws) if ws is not None else 0.0
    wh = _parse_env_scalar(t, "waveh", re.compile(r"浪高[：:\s]*([0-9.]+)\s*米"))
    out["num_wave_height_m_env"] = float(wh) if wh is not None else 0.0
    wp = _parse_env_scalar(
        t, "period", re.compile(r"(?:周期|浪周期)[：:\s]*([0-9.]+)\s*秒")
    )
    out["num_wave_period_s"] = float(wp) if wp is not None else 0.0
    air = _parse_env_scalar(
        t, "air", re.compile(r"(?:气温|空气温度)[：:\s]*([\-0-9.]+)\s*℃")
    )
    out["num_air_temp_c"] = float(air) if air is not None else 0.0
    sea = _parse_env_scalar(
        t, "sea", re.compile(r"(?:水温|表层水温|海水温度)[：:\s]*([\-0-9.]+)")
    )
    out["num_sea_temp_c"] = float(sea) if sea is not None else 0.0
    pr = _parse_env_scalar(
        t, "pres", re.compile(r"(?:气压|海面气压)[：:\s]*([0-9.]+)\s*(?:hPa|百帕)?")
    )
    out["num_pressure_hpa"] = float(pr) if pr is not None else 0.0

    # O_*-style flags / counts (from text only)
    out["flag_mentions_solas"] = 1.0 if "SOLAS" in t.upper() or "索拉斯" in t else 0.0
    out["class_society_keyword_hits"] = float(_count_keywords(t, _CLASS_SOCIETY_KEYS))
    out["num_passengers_reported"] = float(
        max(
            _parse_int_patterns(t, [r"旅客\s*(\d+)", r"载客\s*(\d+)", r"乘客\s*(\d+)"]),
            0,
        )
    )
    out["num_crew_reported_alt"] = float(
        _parse_int_patterns(
            t,
            [r"船员\s*(\d+)\s*人", r"在船船员\s*(\d+)", r"配备船员\s*(\d+)"],
        )
    )

    return out


def engineer_report_features(text: str) -> dict[str, Any]:
    """Rule-based features for tree/linear models + optional text column elsewhere."""
    t = text or ""
    head = t[:50_000]
    features: dict[str, Any] = {
        "env_keyword_hits": float(_count_keywords(t, _ENV_KEYS)),
        "ops_keyword_hits": float(_count_keywords(t, _OPS_KEYS)),
        "vessel_type_hits": float(_count_keywords(t, _VESSEL_KEYS)),
        "human_factor_hits": float(_count_keywords(t, _HUMAN_KEYS)),
        "consequence_hits": float(_count_keywords(t, _CONSEQ_KEYS)),
        "area_keyword_hits": float(_count_keywords(t, _AREA_KEYS)),
        "equipment_keyword_hits": float(_count_keywords(t, _EQUIPMENT_KEYS)),
        "cargo_keyword_hits": float(_count_keywords(t, _CARGO_KEYS)),
        "regulatory_keyword_hits": float(_count_keywords(t, _REGULATORY_KEYS)),
        "pilot_tug_keyword_hits": float(_count_keywords(t, _PILOT_TUG_KEYS)),
    }
    for name, pat in _NUM_PATTERNS:
        v = _parse_first_float(pat, head)
        features[f"num_{name}"] = float(v) if v is not None else 0.0

    wv = _parse_wind_level_max(t)
    features["num_wind_level_max"] = float(wv) if wv is not None else 0.0

    hv = _parse_accident_hour(t)
    features["num_accident_hour"] = float(hv) if hv is not None else -1.0
    if hv is not None:
        h = int(hv)
        night = 1.0 if (h >= 18 or h <= 5) else 0.0
    else:
        night = 0.0
    features["flag_night_accident"] = night

    features.update(_engineer_hint_based_features(t, features))
    return features
