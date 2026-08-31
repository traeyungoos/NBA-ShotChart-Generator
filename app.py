"""
Interactive shot chart app.

Reads any shotdetail CSV or parquet from ./data and lets you pick the file,
season, and player in the browser.

Zone layout follows NBA.com/Stats: the basket sits at the top, and zones
are the cross product of SHOT_ZONE_BASIC x SHOT_ZONE_AREA x SHOT_ZONE_RANGE
folded down to the 14 regions that chart displays.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DATA_GLOBS = ("*.parquet", "*.csv")

NEEDED_COLS = [
    "PLAYER_NAME",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "SHOT_MADE_FLAG",
    "SEASON",
    "GAME_DATE",
]

REQUIRED_COLS = [
    "PLAYER_NAME",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "SHOT_MADE_FLAG",
]

# Court dimensions in tenths of a foot (matches LOC_X / LOC_Y from stats.nba.com)
BASELINE_Y = -52.5
RA_RADIUS = 40.0            # restricted area, 4 ft
R_8FT = 80.0                # "Less Than 8 ft." / "8-16 ft." boundary
R_16FT = 160.0              # "8-16 ft." / "16-24 ft." boundary
THREE_RADIUS = 237.5        # 3pt arc, 23'9"
CORNER_X = 220.0            # straight corner-3 line, 22 ft
CORNER_BREAK_Y = float(np.sqrt(THREE_RADIUS**2 - CORNER_X**2))   # ~89.5
PAINT_HALF_W = 80.0         # paint is 16 ft wide
PAINT_TOP_Y = 137.5         # free throw line
COURT_HALF_W = 250.0
DISPLAY_TOP_Y = 300.0       # crop here; nothing meaningful past the arc

# Angle (deg, CCW from +x axis) where the corner-3 line meets the arc.
# This is the real court: it positions the bend in the drawn 3-point line.
BREAK_DEG = float(np.degrees(np.arcsin(CORNER_BREAK_Y / THREE_RADIUS)))   # ~22.1

# Where stats.nba.com actually stops calling a shot a corner 3. Measured
# from the data, not derived from the court: it is a horizontal cutoff,
# and it sits ~2 tenths of a foot inside the geometric corner above.
# Used for zone polygons; BREAK_DEG is used for drawing. See METHODOLOGY.md.
ZONE_BREAK_Y = 87.5
ZONE_BREAK_DEG = float(np.degrees(np.arcsin(ZONE_BREAK_Y / THREE_RADIUS)))  # ~21.6

# Where the r=160 circle crosses the baseline
X_BL_16 = float(np.sqrt(R_16FT**2 - BASELINE_Y**2))                       # ~151.1
ANG_BL_16 = float(np.degrees(np.arctan2(BASELINE_Y, X_BL_16)))            # ~-19.2

# Area boundaries, read off the data: for every shot compute
# atan2(LOC_Y, LOC_X) in degrees, group by (range band, area label), and
# take the 0.5/99.5 percentiles. Each band's areas tile with no overlap,
# and the boundaries land on round numbers:
#   8-16 ft  band splits at        60 / 120 deg
#   16-24 ft band splits at 36 / 72 / 108 / 144 deg   (= 180/5, a clean fifth)
# The 24+ band is not angular at all — see ZONE_BREAK_Y above.
# Verified identical on NBA 2024-25 (219k shots) and WNBA 2025 (38k).
A_MID = (36.0, 72.0, 108.0, 144.0)
A_NEAR = (60.0, 120.0)

# Colors
BG_COLOR = "#1a1512"
COURT_LINE_COLOR = "#b8a88a"
LABEL_BG = "#1a1512"
CMAP = LinearSegmentedColormap.from_list("rg", ["#7a1f28", "#6a6a60", "#1e8a3c"])
NORM = Normalize(vmin=-0.10, vmax=0.10)
NO_DATA_COLOR = "#413d38"
ZONE_EDGE = "#6b6357"
KEY_LINE_COLOR = "#e6d8bd"

# ---------------------------------------------------------------------------
# Zone definitions
#
# The raw data has ~29 distinct (basic, area, range) triples. NBA.com folds
# them into 14 displayed regions: the low-volume long-corner and sub-24ft
# above-the-break buckets get merged into their neighbours, and the paint
# is shown whole rather than split by range.
# ---------------------------------------------------------------------------

ZONE_LABELS = {
    "RA": "Restricted Area",
    "PAINT": "Paint (Non-RA)",
    "MR_L8": "Mid-Range Left (8-16 ft)",
    "MR_R8": "Mid-Range Right (8-16 ft)",
    "MR_L16": "Mid-Range Left (16-24 ft)",
    "MR_R16": "Mid-Range Right (16-24 ft)",
    "MR_LC": "Mid-Range Left Center",
    "MR_RC": "Mid-Range Right Center",
    "MR_C": "Mid-Range Center",
    "C3_L": "Left Corner 3",
    "C3_R": "Right Corner 3",
    "ATB_LC": "Above the Break 3 (Left)",
    "ATB_C": "Above the Break 3 (Center)",
    "ATB_RC": "Above the Break 3 (Right)",
}


def display_zone(basic, area, rng):
    """Fold a raw (basic, area, range) triple into one displayed region."""
    if basic == "Backcourt" or rng == "Back Court Shot":
        return None
    if basic == "Restricted Area":
        return "RA"
    if basic == "In The Paint (Non-RA)":
        return "PAINT"
    if basic == "Left Corner 3":
        return "C3_L"
    if basic == "Right Corner 3":
        return "C3_R"
    if basic == "Above the Break 3":
        if area in ("Left Side Center(LC)", "Left Side(L)"):
            return "ATB_LC"
        if area in ("Right Side Center(RC)", "Right Side(R)"):
            return "ATB_RC"
        return "ATB_C"
    # Mid-Range
    if area == "Left Side Center(LC)":
        return "MR_LC"
    if area == "Right Side Center(RC)":
        return "MR_RC"
    if area == "Left Side(L)":
        return "MR_L8" if rng == "8-16 ft." else "MR_L16"
    if area == "Right Side(R)":
        return "MR_R8" if rng == "8-16 ft." else "MR_R16"
    return "MR_C"

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def arc_pts(r, deg_start, deg_end, n=70):
    """Points along an arc, degrees CCW from +x axis."""
    a = np.linspace(np.radians(deg_start), np.radians(deg_end), n)
    return list(zip(r * np.cos(a), r * np.sin(a)))


def pie(r, a0, a1, n=70):
    """Wedge from the hoop out to radius r. Inner part gets covered by
    whatever is drawn on top of it, which is how the rings are built."""
    return [(0.0, 0.0)] + arc_pts(r, a0, a1, n)


def mirror(pts):
    """Reflect across the y axis and reverse, preserving winding."""
    return [(-x, y) for x, y in reversed(pts)]

# ---------------------------------------------------------------------------
# Polygons, in draw order (back to front).
#
# Each entry is (zone_id, points). A zone may appear more than once — the
# mid-range centre is two wedges that share a colour and read as one region.
# Drawing outward-in means every ring is trimmed by the one stacked on top,
# so there are no seams to line up by hand.
# ---------------------------------------------------------------------------

def _build_polygons():
    polys = []

    # ---- 24+ ft ----
    # These use ZONE_BREAK_*, the labeling cutoff, not the drawn corner.
    polys.append(("C3_R", [
        (CORNER_X, BASELINE_Y), (COURT_HALF_W, BASELINE_Y),
        (COURT_HALF_W, ZONE_BREAK_Y), (CORNER_X, ZONE_BREAK_Y),
    ]))
    polys.append(("C3_L", mirror(polys[-1][1])))

    x72 = THREE_RADIUS * np.cos(np.radians(72.0))
    atb_r = (
        arc_pts(THREE_RADIUS, ZONE_BREAK_DEG, 72.0)
        + [(x72, DISPLAY_TOP_Y), (COURT_HALF_W, DISPLAY_TOP_Y),
           (COURT_HALF_W, ZONE_BREAK_Y)]
    )
    polys.append(("ATB_RC", atb_r))
    polys.append(("ATB_LC", mirror(atb_r)))
    polys.append(("ATB_C",
        arc_pts(THREE_RADIUS, 72.0, 108.0)
        + [(-x72, DISPLAY_TOP_Y), (x72, DISPLAY_TOP_Y)]
    ))

    # ---- 16-24 ft (wedges out to the arc; trimmed later by the 8-16 ring) ----
    mr_r16 = (
        [(0.0, 0.0), (0.0, BASELINE_Y), (CORNER_X, BASELINE_Y),
         (CORNER_X, ZONE_BREAK_Y)]
        + arc_pts(THREE_RADIUS, ZONE_BREAK_DEG, A_MID[0])
    )
    polys.append(("MR_R16", mr_r16))
    polys.append(("MR_L16", mirror(mr_r16)))
    polys.append(("MR_RC", pie(THREE_RADIUS, A_MID[0], A_MID[1])))
    polys.append(("MR_LC", pie(THREE_RADIUS, A_MID[2], A_MID[3])))
    polys.append(("MR_C", pie(THREE_RADIUS, A_MID[1], A_MID[2])))

    # ---- 8-16 ft (trims the wedges above; trimmed in turn by the paint) ----
    mr_r8 = (
        [(0.0, 0.0), (0.0, BASELINE_Y), (X_BL_16, BASELINE_Y)]
        + arc_pts(R_16FT, ANG_BL_16, A_NEAR[0])
    )
    polys.append(("MR_R8", mr_r8))
    polys.append(("MR_L8", mirror(mr_r8)))
    # Centre 8-16 shares the mid-range centre colour, so the sliver above the
    # free throw line reads continuously with the wedge beyond it.
    polys.append(("MR_C", pie(R_16FT, A_NEAR[0], A_NEAR[1])))

    # ---- paint, then the rim ----
    polys.append(("PAINT", [
        (-PAINT_HALF_W, BASELINE_Y), (PAINT_HALF_W, BASELINE_Y),
        (PAINT_HALF_W, PAINT_TOP_Y), (-PAINT_HALF_W, PAINT_TOP_Y),
    ]))
    polys.append(("RA", arc_pts(RA_RADIUS, 0.0, 360.0, n=100)))

    return polys


POLYGONS = _build_polygons()

# One label per zone, positioned by hand in court coordinates
LABEL_POS = {
    "RA": (0, 8),
    "PAINT": (0, 100),
    "MR_C": (0, 192),
    "MR_LC": (-128, 158),
    "MR_RC": (128, 158),
    "MR_L8": (-116, 26),
    "MR_R8": (116, 26),
    "MR_L16": (-190, 18),
    "MR_R16": (190, 18),
    "C3_L": (-236, 18),
    "C3_R": (236, 18),
    "ATB_LC": (-168, 252),
    "ATB_C": (0, 268),
    "ATB_RC": (168, 252),
}

# ---------------------------------------------------------------------------
# Court lines
# ---------------------------------------------------------------------------

def draw_court(ax):
    lw = 1.5
    c = COURT_LINE_COLOR
    key_lw = 2.6

    ax.plot([-COURT_HALF_W, COURT_HALF_W], [BASELINE_Y, BASELINE_Y], c=c, lw=lw)
    ax.plot([-COURT_HALF_W, -COURT_HALF_W], [BASELINE_Y, DISPLAY_TOP_Y], c=c, lw=lw)
    ax.plot([COURT_HALF_W, COURT_HALF_W], [BASELINE_Y, DISPLAY_TOP_Y], c=c, lw=lw)

    # Paint
    ax.plot([-PAINT_HALF_W, -PAINT_HALF_W], [BASELINE_Y, PAINT_TOP_Y],
            c=KEY_LINE_COLOR, lw=key_lw)
    ax.plot([PAINT_HALF_W, PAINT_HALF_W], [BASELINE_Y, PAINT_TOP_Y],
            c=KEY_LINE_COLOR, lw=key_lw)
    ax.plot([-PAINT_HALF_W, PAINT_HALF_W], [PAINT_TOP_Y, PAINT_TOP_Y],
            c=KEY_LINE_COLOR, lw=key_lw)

    # Free throw circle
    solid = [(x, y + PAINT_TOP_Y) for x, y in arc_pts(60, 0, 180)]
    dash = [(x, y + PAINT_TOP_Y) for x, y in arc_pts(60, 180, 360)]
    ax.plot(*zip(*solid), c=c, lw=lw)
    ax.plot(*zip(*dash), c=c, lw=lw, ls="--")

    # Restricted area
    ax.plot(*zip(*arc_pts(RA_RADIUS, 0, 180)), c=KEY_LINE_COLOR, lw=key_lw)

    # 3-point line
    ax.plot([CORNER_X, CORNER_X], [BASELINE_Y, CORNER_BREAK_Y], c=c, lw=lw)
    ax.plot([-CORNER_X, -CORNER_X], [BASELINE_Y, CORNER_BREAK_Y], c=c, lw=lw)
    ax.plot(*zip(*arc_pts(THREE_RADIUS, BREAK_DEG, 180 - BREAK_DEG, n=110)),
            c=c, lw=lw)

    # Hoop and backboard
    ax.add_patch(plt.Circle((0, 0), 7.5, fill=False, ec=c, lw=lw))
    ax.plot([-30, 30], [-7.5, -7.5], c=c, lw=2)

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def build_chart(zone_df, player_name, subtitle):
    """zone_df: one row per ZONE with fgm, fga, fg_pct, lg_fg, delta."""
    fig, ax = plt.subplots(figsize=(8, 7.6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-COURT_HALF_W - 10, COURT_HALF_W + 10)
    ax.set_ylim(DISPLAY_TOP_Y + 10, BASELINE_Y - 15)
    ax.set_aspect("equal")
    ax.axis("off")

    stats = {row["ZONE"]: row for _, row in zone_df.iterrows()}

    for zone_id, pts in POLYGONS:
        if zone_id in stats:
            delta = np.clip(stats[zone_id]["delta"], -0.10, 0.10)
            color = CMAP(NORM(delta))
        else:
            color = NO_DATA_COLOR
        ax.add_patch(plt.Polygon(pts, fc=color, ec=ZONE_EDGE, lw=0.6, alpha=0.9))

    draw_court(ax)

    for zone_id, (cx, cy) in LABEL_POS.items():
        if zone_id not in stats:
            continue
        row = stats[zone_id]
        ax.text(
            cx, cy, f"{int(row['fgm'])}/{int(row['fga'])}\n{row['fg_pct'] * 100:.1f}%",
            ha="center", va="center", color="white", fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=LABEL_BG, ec="none", alpha=0.9),
        )

    total_fgm = zone_df["fgm"].sum()
    total_fga = zone_df["fga"].sum()
    if total_fga:
        total_pct = total_fgm / total_fga * 100
        expected = (zone_df["fga"] * zone_df["lg_fg"]).sum() / total_fga
        refg = (total_fgm / total_fga - expected) * 100
    else:
        total_pct = refg = 0.0

    ax.set_title(f"{player_name}\n{subtitle}", color="white",
                 fontsize=18, fontweight="bold", pad=12)
    ax.text(
        0.5, -0.02,
        f"{int(total_fgm)}/{int(total_fga)} ({total_pct:.1f}%)    rFG%: {refg:+.1f}%",
        transform=ax.transAxes, ha="center", va="top",
        color="white", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc=BG_COLOR, ec="none"),
    )

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.06, aspect=30)
    cbar.set_label("FG% vs League Avg", color="white", fontsize=9)
    cbar.set_ticks([-0.10, -0.05, 0, 0.05, 0.10])
    cbar.set_ticklabels(["-10%", "-5%", "0%", "+5%", "+10%"])
    cbar.ax.tick_params(colors="white", labelsize=8)

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_data_files():
    if not DATA_DIR.exists():
        return []
    files = []
    for pattern in DATA_GLOBS:
        files.extend(DATA_DIR.glob(pattern))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def file_key(path):
    stat = path.stat()
    return (str(path), stat.st_mtime, stat.st_size)


def _derive_season(game_date):
    """
    Season from GAME_DATE, handling 20230615 and 2023-06-15 alike.

    NBA seasons span two calendar years; WNBA seasons are one. The WNBA
    plays roughly May-October, so testing for a December-March game
    distinguishes them. Testing "September or later" does not — WNBA
    playoffs run into September and October.
    """
    s = game_date.astype(str).str.replace("-", "", regex=False)
    year = pd.to_numeric(s.str[:4], errors="coerce")
    month = pd.to_numeric(s.str[4:6], errors="coerce")
    spans_new_year = ((month == 12) | (month <= 3)).any()
    season = np.where(month >= 9, year, year - 1) if spans_new_year else year
    return pd.Series(season, index=game_date.index).astype("int16")


@st.cache_data(show_spinner="Loading shot data…")
def load_data(key):
    """Read a data file into a compact DataFrame with a ZONE column."""
    path = Path(key[0])

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        available = set(pq.read_schema(path).names)
        df = pd.read_parquet(path, columns=[c for c in NEEDED_COLS if c in available])
    else:
        available = set(pd.read_csv(path, nrows=0).columns)
        df = pd.read_csv(path, usecols=[c for c in NEEDED_COLS if c in available])

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns {missing}")

    if "SEASON" in df.columns:
        df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce").astype("int16")
    elif "GAME_DATE" in df.columns:
        df["SEASON"] = _derive_season(df["GAME_DATE"])
    else:
        raise ValueError("file has neither a SEASON nor a GAME_DATE column")
    df.drop(columns=[c for c in ("GAME_DATE",) if c in df.columns], inplace=True)

    df["SHOT_MADE_FLAG"] = (
        pd.to_numeric(df["SHOT_MADE_FLAG"], errors="coerce").fillna(0).astype("int8")
    )

    # Map the ~29 raw zone triples onto the 14 displayed regions. Done on
    # the unique combos and merged back, so it costs nothing on 800k rows.
    zcols = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE"]
    combos = df[zcols].drop_duplicates()
    combos["ZONE"] = [display_zone(*t) for t in combos.itertuples(index=False)]
    df = df.merge(combos, on=zcols, how="left")

    df = df[df["ZONE"].notna()]          # drops backcourt heaves
    df.drop(columns=zcols, inplace=True)

    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype("category")
    df["ZONE"] = df["ZONE"].astype("category")

    return df.reset_index(drop=True)


@st.cache_data
def season_list(key):
    return sorted(load_data(key)["SEASON"].unique().tolist(), reverse=True)


@st.cache_data
def player_list(key, season):
    df = load_data(key)
    sub = df.loc[df["SEASON"] == season, "PLAYER_NAME"]
    return sorted(sub.cat.remove_unused_categories().cat.categories.tolist())


@st.cache_data
def league_baseline(key, season):
    """League FG% per zone. Touches every shot in the season, so it is
    cached separately — the first player costs ~1s, the rest are instant."""
    df = load_data(key)
    sub = df[df["SEASON"] == season]
    out = (
        sub.groupby("ZONE", observed=True)["SHOT_MADE_FLAG"]
        .agg(lg_fgm="sum", lg_fga="count")
        .reset_index()
    )
    out["lg_fg"] = out["lg_fgm"] / out["lg_fga"]
    return out.drop(columns=["lg_fgm", "lg_fga"])


@st.cache_data
def zone_stats(key, season, player, shrink_k):
    """
    Per-zone stats for one player. shrink_k is a prior strength in
    attempts: a zone gets weighted fga/(fga+k) toward its own rate and
    k/(fga+k) toward league average. 0 gives raw percentages.
    """
    df = load_data(key)
    mask = (df["SEASON"] == season) & (df["PLAYER_NAME"] == player)
    sub = df.loc[mask, ["ZONE", "SHOT_MADE_FLAG"]]
    if sub.empty:
        return pd.DataFrame()

    out = (
        sub.groupby("ZONE", observed=True)["SHOT_MADE_FLAG"]
        .agg(fgm="sum", fga="count")
        .reset_index()
    )
    out["fg_pct"] = out["fgm"] / out["fga"]

    out = out.merge(league_baseline(key, season), on="ZONE", how="left")
    out["lg_fg"] = out["lg_fg"].fillna(out["fg_pct"])

    if shrink_k > 0:
        shrunk = (out["fgm"] + shrink_k * out["lg_fg"]) / (out["fga"] + shrink_k)
        out["delta"] = shrunk - out["lg_fg"]
    else:
        out["delta"] = out["fg_pct"] - out["lg_fg"]

    out["ZONE"] = out["ZONE"].astype(str)
    return out

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Shot Charts", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background-color: #1a1512; }
        .stSelectbox label, .stSlider label, .stMarkdown,
        h1, h2, h3, p { color: #e8e0d4 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    files = list_data_files()
    if not files:
        st.title("Shot Charts")
        st.error(
            f"No data files found in `{DATA_DIR}`.\n\n"
            "Run `python prep_data.py` to download some, or drop any "
            "shotdetail CSV or parquet into that folder."
        )
        return

    with st.sidebar:
        st.header("Data")
        chosen = st.selectbox(
            "File", files,
            format_func=lambda p: f"{p.name}  ({p.stat().st_size / 1e6:.0f} MB)",
        )
        key = file_key(chosen)

        st.header("Options")
        shrink_k = st.slider(
            "Shrinkage", 0, 50, 0, step=5,
            help="Pulls low-volume zones toward league average so a 0/1 "
                 "doesn't paint a zone solid red. The number is a prior "
                 "strength in attempts. 0 = raw percentages.",
        )
        min_att = st.slider(
            "Hide labels below N attempts", 0, 20, 0,
            help="Zones with fewer attempts still get colored, but the "
                 "made/attempted label is suppressed.",
        )

    try:
        seasons = season_list(key)
    except ValueError as e:
        st.title("Shot Charts")
        st.error(f"Couldn't read `{chosen.name}`: {e}")
        return

    st.title("Shot Charts")
    col1, col2 = st.columns([1, 3])
    season = col1.selectbox("Season", seasons)
    player = col2.selectbox("Player", player_list(key, season))
    if not player:
        return

    stats = zone_stats(key, season, player, shrink_k)
    if stats.empty:
        st.warning("No shots found for this player and season.")
        return
    if min_att > 0:
        stats = stats[stats["fga"] >= min_att]
        if stats.empty:
            st.warning(f"No zones with at least {min_att} attempts.")
            return

    fig = build_chart(stats, player, f"{season} · {chosen.stem}")
    left, mid, right = st.columns([1, 3, 1])
    with mid:
        st.pyplot(fig, width="stretch")
    plt.close(fig)

    with st.expander("Zone breakdown"):
        display = stats.copy()
        display["Zone"] = display["ZONE"].map(ZONE_LABELS)
        for col in ("fg_pct", "lg_fg", "delta"):
            display[col] = (display[col] * 100).round(1).astype(str) + "%"
        display = display[["Zone", "fgm", "fga", "fg_pct", "lg_fg", "delta"]].rename(
            columns={"fgm": "Made", "fga": "Attempted", "fg_pct": "FG%",
                     "lg_fg": "League FG%", "delta": "vs League"}
        )
        st.dataframe(display, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()