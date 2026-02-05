# app.py (통합본: [코드1] -> divider -> [코드2])
import os
import time
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

# 코드1
import plotly.express as px
from scipy.stats import linregress, mannwhitneyu

# 코드2
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from shapely.geometry import Point
from scipy.stats import gaussian_kde

# plotly graph_objects (KDE를 plotly로 렌더링)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# 0) Streamlit 기본 설정 (한 번만)
# =========================================================
st.set_page_config(layout="wide", page_title="똘똘한 집 한 채")


# =========================================================
# ✅ (추가) Part 앵커 + 헤더 우측 1/2/3/4 점프 버튼 유틸
# =========================================================
PART_LABELS = {
    "part1": "가격 상승률 vs 거래 변화",
    "part2": "가격-거래량 선행 가능성",
    "part3": "고가 거래의 밀도로 본 돈의 무게중심",
    "part4": "핵심 권역이 만들어지는 7년의 흐름",
    "part5": "돈의 무게중심은 어떻게 이동했나",
}


def _jump_buttons_html(active: str | None = None) -> str:
    """
    헤더 우측에 보이는 1 2 3 4 버튼.
    active: 현재 섹션이면 약간 강조(선택사항)
    """
    items = []
    for i, pid in enumerate(["part1", "part2", "part3", "part4", "part5"], start=1):
        cls = "jump-btn"
        if active == pid:
            cls += " active"
        items.append(f"<a class='{cls}' href='#{pid}' title='{PART_LABELS[pid]}'>{i}</a>")
    return "<div class='jump-wrap'>" + "".join(items) + "</div>"


def anchor(pid: str):
    # 앵커 위치(스크롤 시 살짝 위로 보정)
    st.markdown(f"<div id='{pid}' class='anchor-pad'></div>", unsafe_allow_html=True)


def header_with_jump(title: str, pid: str, level: int = 1):
    """
    level=1: 타이틀급(큰 글자)
    level=2: 섹션 헤더급
    """
    anchor(pid)

    left, right = st.columns([4.5, 1.0], vertical_alignment="center")
    with left:
        if level == 1:
            st.markdown(f"<div class='h1'>{title}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='h2'>{title}</div>", unsafe_allow_html=True)
    with right:
        st.markdown(_jump_buttons_html(active=pid), unsafe_allow_html=True)


# =========================================================
# 1) 레이아웃/CSS
# =========================================================
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none; }
      section.main { margin-left: 0 !important; }
      header[data-testid="stHeader"] { height: 0.5rem; }

      html, body,
      .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main {
            background-color: #FBFBFB !important;
      }

      /* (선택) 상단 헤더/툴바도 배경이 비치게 */
      header[data-testid="stHeader"] { background: transparent !important; }
      [data-testid="stToolbar"] { background: transparent !important; }

      .block-container {
        max-width: 1100px;
        padding-left: 2.6rem;
        padding-right: 2.6rem;
        padding-top: 3.0rem;
        padding-bottom: 2.2rem;
        margin: 0 auto;
      }

      /* --- UI 리듬(8pt) 보정: 너무 큰 selectbox/slider를 살짝만 정리 --- */
      div[data-testid="stSelectbox"] > div,
      div[data-testid="stSlider"] > div {
        margin-top: 0.25rem;
      }
      div[data-testid="stSelectbox"] label,
      div[data-testid="stSlider"] label {
        font-size: 0.85rem;
        color: #6B7280; /* gray-500 */
      }

      /* KDE 4-panel: always show 1x4 */
      .kde-wide { display: block !important; }

      /* ✅ 앵커 이동 보정: Streamlit 상단바(Deploy bar)에 안 가리도록 충분히 위로 */
      .anchor-pad { 
        position: relative; 
        top: -84px;          /* <- 기존 -14px에서 크게 */
        height: 1px;
      }


      /* ✅ 커스텀 헤더 텍스트 (st.title / st.header 대체) */
      .h1 {
        font-size: 1.85rem;   /* 기존 2.05rem -> ↓ */
        font-weight: 760;
        line-height: 1.18;
        color: #111827;
        margin: 0.15rem 0 0.45rem 0;
      }
      .h2 {
        font-size: 1.35rem;   /* 기존 1.50rem -> ↓ */
        font-weight: 740;
        line-height: 1.25;
        color: #111827;
        margin: 0.10rem 0 0.35rem 0;
      }

      /* ✅ Streamlit subheader(h3)도 같은 크기로 */
      div[data-testid="stMarkdownContainer"] h3 {
        font-size: 1.35rem !important;  /* ✅ 통일 */
        font-weight: 740 !important;
        line-height: 1.25 !important;
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
     }


      /* ------------------------------------------------------
       ✅ 헤더 우측 "1 2 3 4" 점프 버튼
      ------------------------------------------------------ */
      .jump-wrap {
        display: flex;
        justify-content: flex-end;
        gap: 8px;
      }

      .jump-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 6px;
        background: #E5E7EB;      /* gray-200 */
        font-weight: 400;
        font-size: 0.95rem;
        user-select: none;
      }

      /* ✅ 밑줄/색상: 링크 상태 전부 강제 */
      .jump-btn,
      .jump-btn:link,
      .jump-btn:visited,
      .jump-btn:hover,
      .jump-btn:active {
        text-decoration: none !important;
        color: #111827 !important;
      }

      /* hover 색 */
      .jump-btn:hover {
        background: #D1D5DB;      /* gray-300 */
      }

      /* ✅ 선택된 버튼(active)도 hover와 동일한 색 */
      .jump-btn.active {
        background: #D1D5DB;      /* gray-300 */
      }

      /* ✅ Streamlit 기본 헤더들(산점도/통계/분석A/B/핵심 권역...) 크기 통일 */
      div[data-testid="stMarkdownContainer"] h3 {   /* st.subheader는 보통 h3로 렌더 */
        font-size: 1.25rem !important;
        font-weight: 740 !important;
        line-height: 1.25 !important;
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
      }

      /* "### 통계" 같이 markdown으로 만든 것도 동일하게 */
      div[data-testid="stMarkdownContainer"] h3 a {
        font-size: inherit !important;
      }
    
      /* ✅ PART5 설명 텍스트 ↔ 차트 사이 간격 */
        .part5-chart-gap {
        height: 25px;   /* 12~32px 사이로 취향대로 조절 */
      }

      
      /* ✅ PART5 맨 아래 추가 여백(스크롤 여유) */
      .part5-bottom-space {
        height: 260px;   /* 필요하면 200~400px 사이로 조절 */
      }
      

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)


# =========================================================
# 2) 폰트 설정: repo의 assets/fonts/NanumGothic.ttf 우선
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
# 3) [코드1] 렌더 함수: 가격 상승률 vs 거래 변화 산점도 + 추가분석
# =========================================================
def render_scatter_section():
    # ✅ PART 1
    header_with_jump(PART_LABELS["part1"], "part1", level=1)

    # -------------------------
    # 0) 데이터 로드
    # -------------------------
    try:
        df = pd.read_csv("district_summary.csv").copy()
    except Exception as e:
        st.error(f"district_summary.csv 로드 실패: {e}")
        return

    df = df[df["trade_count_2023"] > 0].copy()
    df["trade_count_growth"] = (df["trade_count_2025"] - df["trade_count_2023"]) / df["trade_count_2023"]

    # -------------------------
    # 1) 급지 매핑
    # -------------------------
    GRADE_MAP = {
        "강남구": 1, "서초구": 1, "송파구": 1,
        "성동구": 2, "마포구": 2, "강동구": 2, "광진구": 2,
        "동작구": 3, "서대문구": 3, "영등포구": 3, "동대문구": 3,
        "금천구": 4, "강북구": 4, "도봉구": 4, "노원구": 4,
    }
    df["급지"] = df["구명"].map(GRADE_MAP).fillna(0).astype(int)  # 0 = 미분류

    # -------------------------
    # 2) UI 상단
    # -------------------------
    mode = st.radio(
        "지표 모드 선택",
        ["절대지표(거래건수)", "상대지표(거래비중)"],
        horizontal=True,
        key="scatter_mode",
    )

    left, right = st.columns([2.2, 1])

    # -------------------------
    # 3) 급지 선택 (필터링 X, 하이라이트 O)
    # -------------------------
    with right:
        st.subheader("통계")
        grade_opt = st.radio(
            "급지 강조(1~4급지)",
            ["전체", "1급지", "2급지", "3급지", "4급지"],
            horizontal=True,
            key="scatter_grade_opt",
        )

    df_plot = df.copy()
    if grade_opt == "전체":
        df_plot["highlight"] = "전체"
    else:
        g = int(grade_opt.replace("급지", ""))
        df_plot["highlight"] = np.where(df_plot["급지"] == g, f"{g}급지", "기타")

    # -------------------------
    # 4) 산점도 설정
    # -------------------------
    x_col = "price_growth"
    if mode == "절대지표(거래건수)":
        y_col = "trade_count_growth"
        y_label = "거래건수 증가율 (2023→2025)"
    else:
        y_col = "trade_share_change"
        y_label = "거래 비중 변화 (2023→2025)"

    # 버블 사이즈: 정규화 스케일 고정
    size_raw = df_plot["trade_count_2025"].astype(float).clip(lower=1)
    cap_q = np.quantile(size_raw, 0.95)
    size_w = np.minimum(size_raw, cap_q)
    size_norm = (size_w - size_w.min()) / (size_w.max() - size_w.min() + 1e-9)
    df_plot["bubble_size"] = 8 + size_norm * 22  # 8~30

    # 라벨 최소화
    core = ["강남구", "서초구", "송파구"]
    top_x = df_plot.nlargest(3, x_col)["구명"].tolist()
    top_y = df_plot.nlargest(3, y_col)["구명"].tolist()
    bot_y = df_plot.nsmallest(3, y_col)["구명"].tolist()
    label_set = set(core + top_x + top_y + bot_y)
    df_plot["label"] = df_plot["구명"].where(df_plot["구명"].isin(label_set), "")

    # 회귀 통계
    tmp = df_plot[[x_col, y_col]].dropna()
    if len(tmp) >= 3 and tmp[x_col].nunique() > 1:
        lr = linregress(tmp[x_col], tmp[y_col])
        r = lr.rvalue
        beta = lr.slope
        r2 = lr.rvalue ** 2
        p_beta = lr.pvalue
    else:
        r = beta = r2 = p_beta = np.nan

    with right:
        st.markdown(
            f"""
- **상관계수 r:** `{r:.3f}`  
- **회귀 기울기 β:** `{beta:.4f}`  
- **R²:** `{r2:.3f}`  
- **p-value (β):** `{p_beta:.4f}`
            """
        )

    # Hover(툴팁)
    if mode == "절대지표(거래건수)":
        hover_cols = {
            "구명": True,
            "median_price_2023": ":,.0f",
            "median_price_2025": ":,.0f",
            "price_growth": ":.3f",
            "trade_count_2023": ":,",
            "trade_count_2025": ":,",
            "trade_count_growth": ":.2%",
            "trade_share_2023": ":.3f",
            "trade_share_2025": ":.3f",
            "trade_share_change": ":.3f",
            "급지": True,
            "label": False,
            "bubble_size": False,
            "highlight": False,
        }
    else:
        hover_cols = {
            "구명": True,
            "median_price_2023": ":,.0f",
            "median_price_2025": ":,.0f",
            "price_growth": ":.3f",
            "trade_share_2023": ":.3f",
            "trade_share_2025": ":.3f",
            "trade_share_change": ":.3f",
            "trade_count_2023": ":,",
            "trade_count_2025": ":,",
            "trade_count_growth": ":.2%",
            "급지": True,
            "label": False,
            "bubble_size": False,
            "highlight": False,
        }

    # -------------------------
    # 5) 좌측: 산점도 렌더
    # -------------------------
    with left:
        st.subheader("산점도")

        if grade_opt == "전체":
            color_map = {"전체": "#8EC7E8"}
        else:
            g = int(grade_opt.replace("급지", ""))
            color_map = {f"{g}급지": "#E45756", "기타": "#8EC7E8"}

        fig = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            size="bubble_size",
            size_max=30,
            color="highlight",
            color_discrete_map=color_map,
            hover_name="구명",
            text="label",
            trendline="ols",
            labels={x_col: "가격 상승률 (2023→2025)", y_col: y_label, "highlight": "급지 강조"},
            hover_data=hover_cols,
        )
        fig.update_traces(textposition="top center", marker=dict(opacity=0.85))
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    # -------------------------
    # 6) 추가분석: A/B + q 슬라이더
    # -------------------------
    st.divider()

    # ✅ PART 2 (원래 st.header 자리)
    header_with_jump(PART_LABELS["part2"], "part2", level=2)

    st.caption("‘지표 변동이 큰 구 vs 작은 구’로 나눠서 다른 지표의 분포를 비교")

    q = st.slider("상/하위 그룹 분리 분위수 q (상위 q, 하위 1-q)", 0.60, 0.90, 0.70, 0.01, key="scatter_q")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("거래량 변동 상/하위 → 가격 상승률")

        q_hi = df["trade_count_growth"].quantile(q)
        q_lo = df["trade_count_growth"].quantile(1 - q)

        B_high = df[df["trade_count_growth"] >= q_hi].copy()
        B_low = df[df["trade_count_growth"] <= q_lo].copy()

        B_high["group"] = f"거래량 변동 상위(≥{q:.2f})"
        B_low["group"] = f"거래량 변동 하위(≤{1-q:.2f})"
        B_df = pd.concat([B_high, B_low], ignore_index=True)

        figB = px.box(
            B_df,
            x="group",
            y="price_growth",
            points="all",
            labels={"group": "구분", "price_growth": "가격 상승률 (2023→2025)"},
            title="(Boxplot) 거래량 변동 그룹별 가격 상승률",
        )

        # figB 만든 직후
        figB.update_layout(
            title=dict(
                text="(Boxplot) 거래량 변동 그룹별 가격 상승률",
                font=dict(size=14, color="#111827", family=None),
                x=0.0, xanchor="left",
        ),
            title_font=dict(size=14, color="#111827"),  # (호환용)
        )
        # ✅ weight(굵기) 적용: plotly는 title_font에 weight가 직접 안 먹는 경우가 많아서 HTML로 처리
        figB.update_layout(title_text="<b>(Boxplot) 거래량 변동 그룹별 가격 상승률</b>")
        
        st.plotly_chart(figB, width="stretch")

        if len(B_high) > 0 and len(B_low) > 0:
            uB = mannwhitneyu(B_high["price_growth"], B_low["price_growth"], alternative="two-sided")
            st.write(f"- **p-value:** `{uB.pvalue:.4f}`")

    with col2:
        st.subheader("가격 변동 상/하위 → 거래량 증가율")

        p_hi = df["price_growth"].quantile(q)
        p_lo = df["price_growth"].quantile(1 - q)

        A_high = df[df["price_growth"] >= p_hi].copy()
        A_low = df[df["price_growth"] <= p_lo].copy()

        A_high["group"] = f"가격 변동 상위(≥{q:.2f})"
        A_low["group"] = f"가격 변동 하위(≤{1-q:.2f})"
        A_df = pd.concat([A_high, A_low], ignore_index=True)

        figA = px.box(
            A_df,
            x="group",
            y="trade_count_growth",
            points="all",
            labels={"group": "구분", "trade_count_growth": "거래건수 증가율 (2023→2025)"},
            title="(Boxplot) 가격 변동 그룹별 거래량 증가율",
        )

        # figA 만든 직후
        figA.update_layout(
            title=dict(
                text="(Boxplot) 가격 변동 그룹별 거래량 증가율",
                font=dict(size=14, color="#111827", family=None),
                x=0.0, xanchor="left",
            ),
            title_font=dict(size=14, color="#111827"),
        )
        
        figA.update_layout(title_text="<b>(Boxplot) 가격 변동 그룹별 거래량 증가율</b>")


        st.plotly_chart(figA, width="stretch")

        if len(A_high) > 0 and len(A_low) > 0:
            uA = mannwhitneyu(A_high["trade_count_growth"], A_low["trade_count_growth"], alternative="two-sided")
            st.write(f"- **p-value:** `{uA.pvalue:.4f}`")


# =========================================================
# 4) [코드2] (KDE) 섹션
# =========================================================
CSV_PATH_DEFAULT = "아파트실거래가2015_2025.csv"
GEOJSON_URL_DEFAULT = (
    "https://raw.githubusercontent.com/raqoon886/Local_HangJeongDong/master/"
    "hangjeongdong_%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C.geojson"
)

KDE_MODE_FIXED = "top_pct"
KDE_VALUE_LABEL_FIXED = "평당 거래가격"
KDE_VALUE_COL_FIXED = "price_per_pyeong_manwon"

DEFAULT_CAP = 9000
DEFAULT_BW = 0.22
DEFAULT_GRIDSIZE = 250
KDE_TOP_PCT = 0.10
KDE_COLORSCALE = "RdYlBu_r"


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


def kde_Z_single_year(tx_df, gu_boundary, positions, year,
                      value_col="price_manwon", cap=9000, bw=0.22, gridsize=250, top_pct=0.10):
    x, y = synthetic_points_value_weighted_top_pct(
        tx_df, gu_boundary, year, value_col=value_col, top_pct=top_pct, cap=cap
    )
    if len(x) == 0:
        return None
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    return kde(positions).reshape(gridsize, gridsize)


@st.cache_data(show_spinner=True)
def compute_Z_all_years_cached(tx_df: pd.DataFrame, geojson_url: str, years: list[int],
                               value_col: str, cap: int, bw: float, gridsize: int, top_pct: float):
    gu_boundary = load_gu_boundary_from_geojson(geojson_url)
    b3857, positions, extent = prepare_grid_and_boundary_from_url(geojson_url, gridsize)

    Z_by_year = {}
    vmax = 0.0
    for yr in years:
        Z = kde_Z_single_year(
            tx_df, gu_boundary, positions,
            year=int(yr),
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


def make_boundary_traces(b3857: gpd.GeoDataFrame):
    xs, ys = [], []
    for geom in b3857.geometry:
        if geom is None:
            continue
        bd = geom.boundary
        if bd.geom_type == "MultiLineString":
            for line in bd.geoms:
                x, y = line.xy
                xs += list(x) + [None]
                ys += list(y) + [None]
        else:
            x, y = bd.xy
            xs += list(x) + [None]
            ys += list(y) + [None]

    return go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(width=1.2, color="#CBD5E1"),
        opacity=0.85,
        hoverinfo="skip",
        showlegend=False,
    )


def _label_colors_for_year(b: gpd.GeoDataFrame, year: int | None):
    black_set = {"용산구", "강남구", "서초구", "송파구"}
    colors = []
    for gu in b["gu"].tolist():
        c = "black" if gu in black_set else "white"
        if year is not None and int(year) == 2019 and gu == "강남구":
            c = "white"
        colors.append(c)
    return colors


def make_label_trace(b3857: gpd.GeoDataFrame, year: int | None):
    if "cx" not in b3857.columns or "cy" not in b3857.columns:
        b = b3857.copy()
        b["cx"] = b.geometry.centroid.x
        b["cy"] = b.geometry.centroid.y
    else:
        b = b3857

    colors = _label_colors_for_year(b, year)

    return go.Scatter(
        x=b["cx"],
        y=b["cy"],
        mode="text",
        text=b["gu"],
        textposition="middle center",
        textfont=dict(size=10, color=colors),
        hoverinfo="skip",
        showlegend=False,
    )


def heatmap_trace(Z, extent, vmax, colorscale, show_scale, colorbar_x=1.02):
    if Z is None:
        return None

    xmin, xmax, ymin, ymax = extent
    ny, nx = Z.shape
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    cb = dict(
        thickness=16,
        len=0.55,
        x=colorbar_x,
        y=0.5,
        tickfont=dict(size=10, color="#4B5563"),
        title=dict(text="고가 거래 밀도", font=dict(size=12, color="#111827")),
    )

    return go.Heatmap(
        z=Z,
        x=x,
        y=y,
        colorscale=colorscale,
        zmin=0,
        zmax=float(vmax),
        showscale=bool(show_scale),
        colorbar=cb if show_scale else None,
        hovertemplate="고가 거래 밀도=%{z:.3g}<extra></extra>",
    )


def _condition_caption():
    return "각 연도별로 평당 거래가격 상위 10% 거래를 선택해, 구별 평당가격 합을 가중치로 해 점을 생성한 뒤 KDE로 공간 밀도를 추정했습니다."


def _phase_label_for_year(y: int) -> str:
    phase = {
        2019: "집중의 시작 (Point)",
        2020: "확장 전개 (Widening)",
        2021: "확산처럼 보이는 구간 (Pre-zone)",
        2022: "외곽은 식고 중심만 남음 (Cooling)",
        2023: "저점 이후 재집중 (Re-focus)",
        2024: "권역의 윤곽 (Zone forming)",
        2025: "권역의 완성 (Zone)",
    }
    return phase.get(int(y), "")


def render_kde_section():
    # ✅ PART 3
    header_with_jump(PART_LABELS["part3"], "part3", level=1)
    st.caption(_condition_caption())

    path_tx = CSV_PATH_DEFAULT
    geojson_url = GEOJSON_URL_DEFAULT

    try:
        tx = load_tx_csv(path_tx)
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        return

    try:
        _ = load_gu_boundary_from_geojson(geojson_url)
    except Exception as e:
        st.error(f"GeoJSON 로드 실패: {e}")
        return

    years_all = list(range(2019, 2026))

    Z_by_year, vmax_fixed, b3857, extent = compute_Z_all_years_cached(
        tx_df=tx,
        geojson_url=geojson_url,
        years=years_all,
        value_col=KDE_VALUE_COL_FIXED,
        cap=int(DEFAULT_CAP),
        bw=float(DEFAULT_BW),
        gridsize=int(DEFAULT_GRIDSIZE),
        top_pct=float(KDE_TOP_PCT)
    )

    xmin, xmax, ymin, ymax = extent

    # (A) 변화: 2019→2025 흐름(Play)
    st.divider()
    header_with_jump(PART_LABELS["part4"], "part4", level=2)

    st.markdown(
        "<div style='color:#111827; font-size:0.875rem; margin-top:0.15rem;'>"
        "▶︎ 버튼을 누르면, 시간의 흐름에 따라 고가 거래 밀도가 변화하는 모습을 볼 수 있습니다."
        "</div>",
        unsafe_allow_html=True,
    )

    if "kde_playing" not in st.session_state:
        st.session_state.kde_playing = False
    if "kde_year_cur" not in st.session_state:
        st.session_state.kde_year_cur = 2019
    if "kde_speed" not in st.session_state:
        st.session_state.kde_speed = 0.5

    ctrl1, ctrl2, ctrl3 = st.columns([1.0, 2.2, 2.2], vertical_alignment="center")

    with ctrl1:
        c1a, c1b = st.columns(2)
        with c1a:
            if st.button("▶", use_container_width=True, key="kde_btn_play"):
                st.session_state.kde_playing = True
        with c1b:
            if st.button("■", use_container_width=True, key="kde_btn_pause"):
                st.session_state.kde_playing = False

    with ctrl2:
        year_slider = st.slider(
            "연도",
            min_value=2019,
            max_value=2025,
            step=1,
            value=int(st.session_state.kde_year_cur),
            disabled=bool(st.session_state.kde_playing),
            key="kde_year_slider_streamlit",
        )
        if not st.session_state.kde_playing:
            st.session_state.kde_year_cur = int(year_slider)

    with ctrl3:
        SPEED_OPTIONS = [0.05, 0.5, 1.0, 2.0]
        cur = float(st.session_state.get("kde_speed", 0.5))
        nearest = min(SPEED_OPTIONS, key=lambda v: abs(v - cur))

        speed_sec = st.select_slider(
            "재생 속도(초)",
            options=SPEED_OPTIONS,
            value=nearest,
            key="kde_speed_streamlit",
        )
        st.session_state.kde_speed = float(speed_sec)

    st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)

    cur_year = int(st.session_state.kde_year_cur)
    phase_text = _phase_label_for_year(cur_year)
    if phase_text:
        st.markdown(
            f"<div style='margin-top:0.25rem; margin-bottom:0.5rem; color:#111827;'>"
            f"<b>{cur_year}년</b> · <span style='color:#6B7280'>{phase_text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    chart_slot = st.empty()

    def make_single_year_fig(year: int):
        Zy = Z_by_year.get(int(year))
        fig = go.Figure()

        hm = heatmap_trace(Zy, extent, vmax_fixed, KDE_COLORSCALE, show_scale=True, colorbar_x=1.03)
        if hm is not None:
            fig.add_trace(hm)
        else:
            fig.add_annotation(text=f"{int(year)}년<br>데이터 없음", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)

        fig.add_trace(make_boundary_traces(b3857))
        fig.add_trace(make_label_trace(b3857, year=int(year)))

        fig.update_xaxes(range=[xmin, xmax], visible=False, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[ymin, ymax], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)

        fig.update_layout(height=420, margin=dict(l=6, r=6, t=8, b=6))
        return fig

    chart_slot.plotly_chart(
        make_single_year_fig(cur_year),
        width="stretch",
        config={"scrollZoom": True, "displayModeBar": "hover"},
    )

    if st.session_state.kde_playing:
        time.sleep(float(st.session_state.kde_speed))
        if cur_year < 2025:
            st.session_state.kde_year_cur = cur_year + 1
        else:
            st.session_state.kde_playing = False
        st.rerun()

    # (B) 비교: 1x4
    st.divider()

    # ✅ PART 5
    header_with_jump(PART_LABELS["part5"], "part5", level=2)

    st.markdown(
        "<div style='color:#111827; font-size:0.875rem; margin-top:0.15rem;'>"
        "2019·2021·2023·2025를 같은 스케일로 비교해"
        "‘점(Point) → 권역(Zone)’으로의 변화를 확인합니다."
        "</div>",
        unsafe_allow_html=True,
    )

    # ✅ 여기 추가 (텍스트 ↔ 차트 그룹 간격)
    st.markdown("<div class='part5-chart-gap'></div>", unsafe_allow_html=True)


    years_cmp = [2019, 2021, 2023, 2025]

    fig_wide = make_subplots(
        rows=1, cols=4,
        horizontal_spacing=0.02,
        subplot_titles=tuple(
            f"<span style='color:#111827'><b>{y}년</b></span><br>{_phase_label_for_year(y)}"
            for y in years_cmp
        ),
    )

    for a in fig_wide.layout.annotations:
        a.update(xanchor="center", align="center")
        a.font.size = 12

    for i, yr in enumerate(years_cmp, start=1):
        Z = Z_by_year.get(int(yr))
        show_scale = (yr == 2025)

        hm = heatmap_trace(
            Z, extent, vmax_fixed, KDE_COLORSCALE,
            show_scale=show_scale,
            colorbar_x=1.02
        )

        if hm is not None:
            fig_wide.add_trace(hm, row=1, col=i)
        else:
            fig_wide.add_annotation(
                text=f"{int(yr)}년<br>데이터 없음",
                x=0.5, y=0.5,
                xref=f"x{i} domain",
                yref=f"y{i} domain",
                showarrow=False,
            )

        fig_wide.add_trace(make_boundary_traces(b3857), row=1, col=i)
        fig_wide.add_trace(make_label_trace(b3857, year=int(yr)), row=1, col=i)

        fig_wide.update_xaxes(range=[xmin, xmax], visible=False, showgrid=False, zeroline=False, row=1, col=i)
        fig_wide.update_yaxes(
            range=[ymin, ymax],
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor=f"x{i}" if i != 1 else "x",
            scaleratio=1,
            row=1, col=i
        )

    for i in [2, 3, 4]:
        fig_wide.update_xaxes(matches="x", row=1, col=i)
        fig_wide.update_yaxes(matches="y", row=1, col=i)

    fig_wide.update_layout(height=380, margin=dict(l=6, r=6, t=48, b=6))

    st.plotly_chart(
        fig_wide,
        width="stretch",
        config={"scrollZoom": True, "displayModeBar": "hover"},
    )
    
    # ✅ PART5 하단 여백(스크롤 여유) 추가
    st.markdown("<div class='part5-bottom-space'></div>", unsafe_allow_html=True)


# =========================================================
# 6) 실행: [코드1] 다음에 [코드2]
# =========================================================
render_scatter_section()
st.divider()
render_kde_section()
