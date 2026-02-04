# app.py
import os
import time
import hashlib

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib import font_manager
from shapely.geometry import Point
from scipy.stats import gaussian_kde


# =========================================================
# 0) Streamlit 기본 설정
# =========================================================
st.set_page_config(page_title="Seoul KDE", layout="wide")


# =========================================================
# 1) 레이아웃: 좌우 여백 + 사이드바 숨김 + 상단 여백(잘림 방지)
# =========================================================
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none; }
      section.main { margin-left: 0 !important; }

      header[data-testid="stHeader"] { height: 0.5rem; }

      .block-container {
        max-width: 1100px;
        padding-left: 2.6rem;
        padding-right: 2.6rem;
        padding-top: 3.0rem;
        padding-bottom: 2.2rem;
        margin: 0 auto;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)


# =========================================================
# 2) 폰트 설정: repo에 포함된 나눔고딕 우선
# =========================================================
def set_korean_font_for_cloud():
    font_path = os.path.join("assets", "fonts", "NanumGothic.ttf")
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        try:
            font_manager._rebuild()
        except Exception:
            pass
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        mpl.rcParams["font.family"] = font_name
    else:
        mpl.rcParams["font.family"] = ["NanumGothic", "Noto Sans CJK KR", "AppleGothic", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False


set_korean_font_for_cloud()


# =========================================================
# 3) 내부 고정 설정 (페이지에 안 보이게)
# =========================================================
CSV_PATH_DEFAULT = "아파트실거래가2015_2025.csv"
GEOJSON_URL_DEFAULT = (
    "https://raw.githubusercontent.com/raqoon886/Local_HangJeongDong/master/"
    "hangjeongdong_%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C.geojson"
)

VALUE_LABEL_TO_COL = {
    "거래가격": "price_manwon",
    "평당 거래가격": "price_per_pyeong_manwon",
}


def mode_label(mode: str) -> str:
    return "전체 거래" if mode == "raw" else "상위 10%"


def make_condition_text_simple(mode: str, value_label: str, bw: float) -> str:
    return f"{mode_label(mode)} 기준 ・ 값 컬럼: {value_label} ・ bw: {bw:.2f}"


# =========================================================
# 4) 유틸
# =========================================================
def stable_seed(*parts) -> int:
    s = "|".join(map(str, parts))
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def sample_points_within_polygon(poly, n, seed=42, max_iter=250000):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = poly.bounds
    points, it = [], 0
    while len(points) < n and it < max_iter:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if poly.contains(p):
            points.append(p)
        it += 1
    return points


def add_gu_labels_plain(ax, b3857, fontsize=8, color="black"):
    for _, r in b3857.iterrows():
        ax.text(r["cx"], r["cy"], r["gu"], ha="center", va="center", fontsize=fontsize, color=color)


def add_gu_labels_selective_black(
    ax, b3857, year=None,
    black_gus=("용산구", "강남구", "서초구", "송파구"),
    fontsize=8,
    color_black="black",
    color_other="white",
    override_white=None
):
    black_set = set(black_gus)
    override_white = override_white or set()

    for _, row in b3857.iterrows():
        cx, cy = float(row["cx"]), float(row["cy"])
        gu = row["gu"]
        label_color = color_black if gu in black_set else color_other
        if year is not None and (year, gu) in override_white:
            label_color = color_other
        ax.text(cx, cy, gu, ha="center", va="center", fontsize=fontsize, color=label_color, zorder=5)


# =========================================================
# 5) 데이터 로드 / boundary / grid 캐시
# =========================================================
@st.cache_data(show_spinner=True)
def load_tx_csv(path_tx: str) -> pd.DataFrame:
    tx = pd.read_csv(path_tx, encoding="utf-8", low_memory=False)

    tx["price_manwon"] = (
        tx["dealAmount"].astype(str)
          .str.replace(",", "", regex=False)
          .str.replace(r"[^0-9.]", "", regex=True)
    )
    tx["price_manwon"] = pd.to_numeric(tx["price_manwon"], errors="coerce")

    tx["area_m2"] = pd.to_numeric(tx["excluUseAr"], errors="coerce")
    tx["price_per_pyeong_manwon"] = tx["price_manwon"] / tx["area_m2"] * 3.3

    tx["ym"] = pd.to_datetime(tx["연월"].astype(str) + "-01", errors="coerce")
    tx["year"] = pd.to_numeric(tx["dealYear"], errors="coerce").astype("Int64")
    tx["gu"] = tx["구명"].astype(str).str.strip()

    tx = tx.dropna(subset=["ym", "year", "gu"])
    tx = tx[(tx["year"] >= 2019) & (tx["year"] <= 2025)].copy()
    return tx


@st.cache_data(show_spinner=True)
def load_gu_boundary_from_geojson(url_geojson: str) -> gpd.GeoDataFrame:
    dong = gpd.read_file(url_geojson)
    if "adm_nm" not in dong.columns:
        raise ValueError("'adm_nm' 컬럼이 없습니다.")
    dong["gu"] = dong["adm_nm"].astype(str).str.extract(r"서울특별시\s+([^\s]+구)")
    gu_boundary = dong.dropna(subset=["gu"]).dissolve(by="gu", as_index=False)

    if gu_boundary.crs is None:
        gu_boundary = gu_boundary.set_crs(epsg=4326)
    gu_boundary = gu_boundary.to_crs(epsg=4326)
    return gu_boundary


@st.cache_data(show_spinner=False)
def prepare_grid_and_boundary_from_url(url_geojson: str, gridsize: int):
    gu_boundary = load_gu_boundary_from_geojson(url_geojson)

    b3857 = gu_boundary.to_crs(epsg=3857).copy()
    b3857["cx"] = b3857.geometry.centroid.x
    b3857["cy"] = b3857.geometry.centroid.y
    xmin, ymin, xmax, ymax = b3857.total_bounds

    X, Y = np.meshgrid(np.linspace(xmin, xmax, gridsize),
                       np.linspace(ymin, ymax, gridsize))
    positions = np.vstack([X.ravel(), Y.ravel()])
    extent = (xmin, xmax, ymin, ymax)
    return b3857, positions, extent


# =========================================================
# 6) 합성점 / KDE
# =========================================================
def synthetic_points_value_weighted_for_year(tx_df, gu_boundary, year, cap=9000, value_col="price_manwon"):
    b = gu_boundary.to_crs(epsg=3857).copy()
    dfy = tx_df[tx_df["year"] == year].copy()

    v = pd.to_numeric(dfy[value_col], errors="coerce")
    dfy = dfy.assign(_v=v).dropna(subset=["gu", "_v"])
    v_by_gu = dfy.groupby("gu")["_v"].sum()

    v_arr = v_by_gu.reindex(b["gu"]).fillna(0.0).values
    total_v = float(np.nansum(v_arr))
    if total_v <= 0:
        return np.array([]), np.array([]), 0.0

    share = v_arr / total_v
    n_dot = np.round(share * cap).astype(int)

    pts = []
    for i, gu in enumerate(b["gu"].values):
        k = int(n_dot[i])
        if k <= 0:
            continue
        poly = b.loc[b["gu"] == gu, "geometry"].values[0]
        seed = stable_seed(year, gu, "raw")
        pts.extend(sample_points_within_polygon(poly, k, seed=seed))

    xs = np.array([p.x for p in pts], dtype=float)
    ys = np.array([p.y for p in pts], dtype=float)
    return xs, ys, total_v


def filter_top_percent_trades(tx_df, year, value_col="price_manwon", top_pct=0.10):
    d = tx_df[tx_df["year"] == year].copy()
    if len(d) == 0:
        return d
    d = d.dropna(subset=[value_col, "gu"])
    if len(d) == 0:
        return d
    thr = d[value_col].quantile(1 - top_pct)
    return d[d[value_col] >= thr]


def synthetic_points_value_weighted_top_pct(tx_df, gu_boundary, year, value_col="price_manwon", top_pct=0.10, cap=9000):
    d = filter_top_percent_trades(tx_df, year, value_col=value_col, top_pct=top_pct)
    if len(d) == 0:
        return np.array([]), np.array([])

    b = gu_boundary.to_crs(epsg=3857).copy()
    value_by_gu = d.groupby("gu")[value_col].sum()
    value_by_gu = value_by_gu.reindex(b["gu"]).fillna(0.0)

    total_value = float(value_by_gu.sum())
    if total_value <= 0:
        return np.array([]), np.array([])

    n_dot_arr = (value_by_gu / total_value * cap).round().astype(int).values

    pts = []
    for i, gu in enumerate(b["gu"].values):
        k = int(n_dot_arr[i])
        if k <= 0:
            continue
        poly = b.loc[b["gu"] == gu, "geometry"].values[0]
        seed = stable_seed(year, gu, "top10")
        pts.extend(sample_points_within_polygon(poly, k, seed=seed))

    xs = np.array([p.x for p in pts], dtype=float)
    ys = np.array([p.y for p in pts], dtype=float)
    return xs, ys


def kde_Z_single_year(tx_df, gu_boundary, positions, year, mode="top_pct",
                      value_col="price_manwon", cap=9000, bw=0.22, gridsize=250, top_pct=0.10):
    if mode == "raw":
        x, y, _ = synthetic_points_value_weighted_for_year(tx_df, gu_boundary, year, cap=cap, value_col=value_col)
        if len(x) == 0:
            return None
        kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
        return kde(positions).reshape(gridsize, gridsize)

    x, y = synthetic_points_value_weighted_top_pct(tx_df, gu_boundary, year, value_col=value_col, top_pct=top_pct, cap=cap)
    if len(x) == 0:
        return None
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    return kde(positions).reshape(gridsize, gridsize)


@st.cache_data(show_spinner=True)
def compute_Z_all_years_cached(tx_df: pd.DataFrame, geojson_url: str, years: list[int],
                               mode: str, value_col: str, cap: int, bw: float, gridsize: int, top_pct: float):
    gu_boundary = load_gu_boundary_from_geojson(geojson_url)
    b3857, positions, extent = prepare_grid_and_boundary_from_url(geojson_url, gridsize)

    Z_by_year = {}
    vmax = 0.0
    for yr in years:
        Z = kde_Z_single_year(
            tx_df, gu_boundary, positions,
            year=int(yr),
            mode=mode,
            value_col=value_col,
            cap=cap,
            bw=bw,
            gridsize=gridsize,
            top_pct=top_pct
        )
        Z_by_year[int(yr)] = Z
        if Z is not None:
            vmax = max(vmax, float(np.nanmax(Z)))

    if vmax <= 0:
        vmax = 1e-12

    return Z_by_year, vmax, b3857, extent


# =========================================================
# 7) Plot
# - 비교: 각 패널 상단 중앙에 연도 표시(요청)
# =========================================================
def plot_two_panel_compare_from_cache(Z_by_year, vmax, b3857, extent, year_left, year_right, mode, cmap):
    xmin, xmax, ymin, ymax = extent

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.02)

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    axes = [axL, axR]
    years = [int(year_left), int(year_right)]

    last_im = None
    for ax, yr in zip(axes, years):
        ax.set_axis_off()
        Z = Z_by_year.get(int(yr))

        if Z is None:
            ax.text(0.5, 0.5, f"{yr}\n데이터 없음",
                    transform=ax.transAxes, ha="center", va="center", color="black")
            continue

        last_im = ax.imshow(
            Z, origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            cmap=cmap, vmin=0, vmax=vmax, alpha=0.95
        )
        b3857.boundary.plot(ax=ax, linewidth=0.5, color="lightgray", alpha=0.9)

        if mode == "top_pct":
            add_gu_labels_selective_black(
                ax, b3857, year=yr,
                black_gus=("용산구", "강남구", "서초구", "송파구"),
                fontsize=8,
                color_black="black",
                color_other="white",
                override_white={(2019, "강남구")}
            )
        else:
            add_gu_labels_plain(ax, b3857, fontsize=8, color="black")

        # ✅ (반영) 각 차트 위 가운데에 연도 표시
        ax.text(
            0.5, 1.02, f"{int(yr)}년",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, color="black"
        )

    if last_im is not None:
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("Value-weighted KDE", fontsize=10)
    else:
        cax.set_axis_off()

    return fig


def plot_single_year_from_cache(Z_by_year, vmax, b3857, extent, year, mode, cmap):
    xmin, xmax, ymin, ymax = extent
    Z = Z_by_year.get(int(year))

    fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.07], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    ax.set_axis_off()

    if Z is None:
        ax.text(0.5, 0.5, f"{year}\n데이터 없음",
                transform=ax.transAxes, ha="center", va="center", color="black")
        cax.set_axis_off()
        return fig

    im = ax.imshow(
        Z, origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap, vmin=0, vmax=vmax, alpha=0.95
    )
    b3857.boundary.plot(ax=ax, linewidth=0.5, color="lightgray", alpha=0.9)

    if mode == "top_pct":
        add_gu_labels_selective_black(
            ax, b3857, year=int(year),
            black_gus=("용산구", "강남구", "서초구", "송파구"),
            fontsize=8,
            color_black="black",
            color_other="white",
            override_white={(2019, "강남구")}
        )
    else:
        add_gu_labels_plain(ax, b3857, fontsize=8, color="black")

    ax.text(0.5, 1.02, f"{int(year)}년",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color="black")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Value-weighted KDE", fontsize=10)

    return fig


# =========================================================
# 8) UI 옵션
# - 옵션 파트 전체를 "하나의 토글"로 묶기(요청)
# - 각 옵션 위젯 자체(라디오/슬라이더)는 기존 코드 유지
# =========================================================
DEFAULT_CAP = 9000
DEFAULT_BW = 0.22
DEFAULT_GRIDSIZE = 250

path_tx = CSV_PATH_DEFAULT
geojson_url = GEOJSON_URL_DEFAULT

show_options = st.toggle("옵션", value=False)

if show_options:
    row1a, row1b, row1c = st.columns([1.4, 1.4, 1.4], vertical_alignment="center")

    with row1a:
        mode_ui = st.radio("모드", ["전체 거래", "상위 10%"], index=1, horizontal=True, key="mode_ui")
        mode = "raw" if mode_ui == "전체 거래" else "top_pct"

    with row1b:
        value_label = st.radio("값 컬럼", list(VALUE_LABEL_TO_COL.keys()), index=1, horizontal=True, key="value_label_ui")
        value_col = VALUE_LABEL_TO_COL[value_label]

    with row1c:
        st.empty()

    row2a, row2b, row2c = st.columns([1.4, 1.4, 1.4], vertical_alignment="center")

    with row2a:
        cap = st.slider("cap(합성점 수)", min_value=1000, max_value=20000, value=int(DEFAULT_CAP), step=500, key="cap_ui")

    with row2b:
        bw = st.slider("bw(bandwidth)", min_value=0.05, max_value=1.00, value=float(DEFAULT_BW), step=0.01, key="bw_ui")

    with row2c:
        gridsize = st.slider("gridsize", min_value=120, max_value=400, value=int(DEFAULT_GRIDSIZE), step=10, key="gridsize_ui")

    st.divider()
else:
    # 옵션을 숨겨도 기존 선택값(있으면)을 유지, 없으면 기본값 사용
    mode_ui = st.session_state.get("mode_ui", "상위 10%")
    mode = "raw" if mode_ui == "전체 거래" else "top_pct"

    value_label = st.session_state.get("value_label_ui", "평당 거래가격")
    if value_label not in VALUE_LABEL_TO_COL:
        value_label = "평당 거래가격"
    value_col = VALUE_LABEL_TO_COL[value_label]

    cap = int(st.session_state.get("cap_ui", DEFAULT_CAP))
    bw = float(st.session_state.get("bw_ui", DEFAULT_BW))
    gridsize = int(st.session_state.get("gridsize_ui", DEFAULT_GRIDSIZE))


# =========================================================
# 9) 데이터 로드 + KDE 미리 계산
# =========================================================
try:
    tx = load_tx_csv(path_tx)
except Exception as e:
    st.error(f"CSV 로드 실패: {e}")
    st.stop()

try:
    _ = load_gu_boundary_from_geojson(geojson_url)
except Exception as e:
    st.error(f"GeoJSON 로드 실패: {e}")
    st.stop()

years_all = list(range(2019, 2026))
cmap = plt.cm.RdYlBu_r
top_pct = 0.10

Z_by_year, vmax_fixed, b3857, extent = compute_Z_all_years_cached(
    tx_df=tx,
    geojson_url=geojson_url,
    years=years_all,
    mode=mode,
    value_col=value_col,
    cap=int(cap),
    bw=float(bw),
    gridsize=int(gridsize),
    top_pct=top_pct
)


# =========================================================
# 10) 거래 금액 가중 KDE 비교
# =========================================================
st.header("거래 금액 가중 KDE 비교")
st.caption(make_condition_text_simple(mode=mode, value_label=value_label, bw=float(bw)))

padL, colL, gap, colR, padR = st.columns([1.2, 2.0, 0.8, 2.0, 1.2], vertical_alignment="center")
with colL:
    year_left = st.selectbox("연도(좌)", years_all, index=0, key="year_left_compare")
with colR:
    year_right = st.selectbox("연도(우)", years_all, index=len(years_all) - 1, key="year_right_compare")

fig_compare = plot_two_panel_compare_from_cache(
    Z_by_year=Z_by_year,
    vmax=vmax_fixed,
    b3857=b3857,
    extent=extent,
    year_left=int(year_left),
    year_right=int(year_right),
    mode=mode,
    cmap=cmap
)
st.pyplot(fig_compare, clear_figure=True)

st.divider()


# =========================================================
# 11) 거래 금액 가중 KDE의 변화 (Play + Slider)
# =========================================================
st.header("거래 금액 가중 KDE의 변화")
st.caption(make_condition_text_simple(mode=mode, value_label=value_label, bw=float(bw)))

# ---- session_state init ----
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

if "anim_speed_widget" not in st.session_state:
    st.session_state.anim_speed_widget = 0.25

if "anim_year_cur" not in st.session_state:
    st.session_state.anim_year_cur = 2019

if "anim_year_slider" not in st.session_state:
    st.session_state.anim_year_slider = int(st.session_state.anim_year_cur)

if "anim_year_next" in st.session_state:
    nxt = int(st.session_state.anim_year_next)
    st.session_state.anim_year_cur = nxt
    st.session_state.anim_year_slider = nxt
    del st.session_state["anim_year_next"]

ctrl1, ctrl2, ctrl3 = st.columns([1.0, 2.0, 2.0], vertical_alignment="center")

with ctrl1:
    if st.button("▶ Play", use_container_width=True):
        st.session_state.is_playing = True
        st.session_state.anim_year_slider = int(st.session_state.anim_year_cur)

with ctrl2:
    st.slider(
        "연도",
        min_value=2019,
        max_value=2025,
        step=1,
        key="anim_year_slider",
        disabled=bool(st.session_state.is_playing),
    )
    if not st.session_state.is_playing:
        st.session_state.anim_year_cur = int(st.session_state.anim_year_slider)

with ctrl3:
    st.slider(
        "재생 속도(초)",
        min_value=0.05,
        max_value=1.00,
        step=0.05,
        key="anim_speed_widget",
    )

current_year = int(st.session_state.anim_year_cur)
speed = float(st.session_state.anim_speed_widget)

fig_single = plot_single_year_from_cache(
    Z_by_year=Z_by_year,
    vmax=vmax_fixed,
    b3857=b3857,
    extent=extent,
    year=current_year,
    mode=mode,
    cmap=cmap
)
st.pyplot(fig_single, clear_figure=True)

if st.session_state.is_playing:
    time.sleep(speed)

    if current_year < 2025:
        st.session_state.anim_year_next = current_year + 1
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.rerun()
