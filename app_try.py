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
# 1) 레이아웃: [코드2] 스타일을 전체 페이지에 적용
#  - (추가) selectbox/slider 높이/여백을 "무겁지 않게" 약간만 정리
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

      /* ======================================================
         ✅ KDE 비교 라디오 전용 스타일 (id wrapper로 범위 제한)
         - 라디오 라벨(“비교 연도 선택”): 짙은 회색
         - 라디오 옵션 텍스트: 기본 짙은 회색
         - 선택된 옵션 텍스트만 검정
      ====================================================== */
      #kde-compare-radio div[data-testid="stRadio"] > label {
        color: #374151 !important; /* gray-700 */
      }

      /* 옵션 텍스트: 기본값 */
      #kde-compare-radio input[type="radio"] + div {
        color: #374151 !important; /* gray-700 */
      }

      /* 선택된 옵션 텍스트만 검정 */
      #kde-compare-radio input[type="radio"]:checked + div {
        color: #111827 !important; /* gray-900 */
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
    st.title("가격 상승률 vs 거래 변화 (통합 시각화)")

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
        "강남구": 1, "서초구": 1, "송파구": 1, "용산구": 1,
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
        st.markdown("### 통계")
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
            color_map = {"전체": "#4C78A8"}
        else:
            g = int(grade_opt.replace("급지", ""))
            color_map = {f"{g}급지": "#E45756", "기타": "#4C78A8"}

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
    st.header("추가분석: 가격 ↔ 거래량 선행 가능성(분석 A/B)")
    st.caption("‘지표 변동이 큰 구 vs 작은 구’로 나눠서 다른 지표의 분포를 비교")

    q = st.slider("상/하위 그룹 분리 분위수 q (상위 q, 하위 1-q)", 0.60, 0.90, 0.70, 0.01, key="scatter_q")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("분석 A: 거래량 변동 상/하위 → 가격 상승률")

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
        st.plotly_chart(figB, width="stretch")

        if len(B_high) > 0 and len(B_low) > 0:
            uB = mannwhitneyu(B_high["price_growth"], B_low["price_growth"], alternative="two-sided")
            st.write(f"- **p-value:** `{uB.pvalue:.4f}`")

    with col2:
        st.subheader("분석 B: 가격 변동 상/하위 → 거래량 증가율")

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
        st.plotly_chart(figA, width="stretch")

        if len(A_high) > 0 and len(A_low) > 0:
            uA = mannwhitneyu(A_high["trade_count_growth"], A_low["trade_count_growth"], alternative="two-sided")
            st.write(f"- **p-value:** `{uA.pvalue:.4f}`")


# =========================================================
# 4) [코드2] (KDE) 섹션: 폴더 구조/파일명 반영
#   ✅ 발표/시연용 UI/UX 반영 버전
#   - "옵션" 제거 (하나의 강한 관점 고정)
#   - 비교: (과거 연도) vs (2025 고정)
#   - 변화: Play/연도/속도 + "해석 가이드" + 연도별 상태 문장
#   - 컬러바: 숫자 중심 → 해석 중심(자본 밀도)
# =========================================================
CSV_PATH_DEFAULT = "아파트실거래가2015_2025.csv"
GEOJSON_URL_DEFAULT = (
    "https://raw.githubusercontent.com/raqoon886/Local_HangJeongDong/master/"
    "hangjeongdong_%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C.geojson"
)

# ✅ 발표용 고정 설정(옵션 제거)
# - 상위 10% 거래
# - 값 컬럼: (평당 거래가격) 권장
KDE_MODE_FIXED = "top_pct"                  # "raw" 또는 "top_pct" 중 발표용은 top_pct 고정
KDE_VALUE_LABEL_FIXED = "평당 거래가격"      # 라벨용(캡션)
KDE_VALUE_COL_FIXED = "price_per_pyeong_manwon"  # 실제 계산 컬럼

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


# =========================
# Plotly 렌더 유틸: 경계선 + 라벨(halo 제거)
# =========================
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

    # ✅ 발표용: 선을 조금 더 “지도처럼” 보이게(너무 얇으면 heat에 묻힘)
    return go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(width=1.2, color="#CBD5E1"),  # slate-300
        opacity=0.85,
        hoverinfo="skip",
        showlegend=False,
    )


def _label_colors_for_year(b: gpd.GeoDataFrame, year: int | None):
    # ✅ heat 위 라벨 가독성(발표용) : 핵심권역은 검정, 나머지는 흰색
    black_set = {"용산구", "강남구", "서초구", "송파구"}
    colors = []
    for gu in b["gu"].tolist():
        c = "black" if gu in black_set else "white"
        # 2019 강남구는 배경에 따라 흰색이 더 안전했던 케이스 유지
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
        textfont=dict(size=11, color=colors),
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

    # ✅ 발표용 컬러바: 숫자보다 “의미”를 먼저 보이게
    cb = dict(
        thickness=18,
        len=0.70,
        x=colorbar_x,
        y=0.5,
        tickfont=dict(size=10, color="#4B5563"),
        title=dict(text="자본 밀도", font=dict(size=12, color="#111827")),
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
        hovertemplate="자본 밀도=%{z:.3g}<extra></extra>",
    )


def _condition_caption():
    # ✅ 발표용: 방법을 짧고 자연스럽게 설명
    return f"상위 10% 거래에 대해 {KDE_VALUE_LABEL_FIXED}을 기준으로 계산한 KDE(Kernel Density Estimation) 지도"


def _compare_narrative_for_year(y: int) -> str:
    # ✅ 비교 파트(연도 vs 2025) 설명 문구
    msg = {
        2019: "2019년엔 ‘점’처럼 모였던 자본이, 2025년엔 ‘권역’으로 연결됩니다.",
        2021: "2021년의 확산은 ‘분산’이 아니라, 권역이 만들어지는 전조였습니다.",
        2023: "2023년 저점 이후, 자본은 서울 전반이 아니라 핵심 권역으로 직행합니다.",
    }
    return msg.get(int(y), "과거의 돈이 2025년에 어떻게 ‘권역’으로 굳어졌는지 비교합니다.")


def _phase_label_for_year(y: int) -> str:
    # ✅ 변화 파트(연속 재생) 상태 문장
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


# =========================================================
# 5) KDE 섹션 (발표/시연용)
# =========================================================
def render_kde_section():
    st.header("거래 금액으로 본 자본의 무게중심 지도")
    st.caption(_condition_caption())

    path_tx = CSV_PATH_DEFAULT
    geojson_url = GEOJSON_URL_DEFAULT

    # 데이터 로드 + KDE 계산
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

    # -----------------------------------------------------
    # (A) 비교: "각 연도 vs 2025(고정)"
    # -----------------------------------------------------
    st.divider()
    st.subheader("돈의 무게중심은 어디로 이동했나")

    # ✅ caption 대신 검은색 텍스트로
    st.markdown(
        "<div style='color:#111827; font-size:0.875rem; margin-top:0.15rem;'>"
        "2025년을 기준 시점으로 고정하고, 과거의 ‘점’이 어떻게 ‘권역’이 되었는지 비교합니다."
        "</div>",
        unsafe_allow_html=True,
    )

    # ✅ 비교 연도 선택 라디오를 wrapper로 감싸서 “이 라디오만” CSS 적용
    st.markdown("<div id='kde-compare-radio'>", unsafe_allow_html=True)
    year_left = st.radio(
        "비교 연도 선택",
        options=[2019, 2021, 2023],
        horizontal=True,
        key="kde_compare_left_year_radio",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ✅ 2025는 화면에 표시하지 않고 내부 기준값으로만 고정
    year_right = 2025

    st.info(_compare_narrative_for_year(int(year_left)))

    ZL = Z_by_year.get(int(year_left))
    ZR = Z_by_year.get(int(year_right))

    fig_cmp = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.03,
        subplot_titles=(f"{int(year_left)}년", f"{int(year_right)}년"),
    )

    hmL = heatmap_trace(ZL, extent, vmax_fixed, KDE_COLORSCALE, show_scale=False)
    hmR = heatmap_trace(ZR, extent, vmax_fixed, KDE_COLORSCALE, show_scale=True, colorbar_x=1.03)

    if hmL is not None:
        fig_cmp.add_trace(hmL, row=1, col=1)
    else:
        fig_cmp.add_annotation(text=f"{int(year_left)}년<br>데이터 없음", x=0.22, y=0.5, xref="paper", yref="paper", showarrow=False)

    if hmR is not None:
        fig_cmp.add_trace(hmR, row=1, col=2)
    else:
        fig_cmp.add_annotation(text=f"{int(year_right)}년<br>데이터 없음", x=0.78, y=0.5, xref="paper", yref="paper", showarrow=False)

    # 경계선 + 라벨
    fig_cmp.add_trace(make_boundary_traces(b3857), row=1, col=1)
    fig_cmp.add_trace(make_boundary_traces(b3857), row=1, col=2)
    fig_cmp.add_trace(make_label_trace(b3857, year=int(year_left)), row=1, col=1)
    fig_cmp.add_trace(make_label_trace(b3857, year=int(year_right)), row=1, col=2)

    # 줌 연동
    fig_cmp.update_xaxes(matches="x", row=1, col=2)
    fig_cmp.update_yaxes(matches="y", row=1, col=2)

    # range 고정
    fig_cmp.update_xaxes(range=[xmin, xmax], visible=False, showgrid=False, zeroline=False, row=1, col=1)
    fig_cmp.update_yaxes(range=[ymin, ymax], visible=False, showgrid=False, zeroline=False,
                         scaleanchor="x", scaleratio=1, row=1, col=1)
    fig_cmp.update_xaxes(range=[xmin, xmax], visible=False, showgrid=False, zeroline=False, row=1, col=2)
    fig_cmp.update_yaxes(range=[ymin, ymax], visible=False, showgrid=False, zeroline=False,
                         scaleanchor="x2", scaleratio=1, row=1, col=2)

    fig_cmp.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig_cmp,
        width="stretch",
        config={
            "scrollZoom": True,
            "displayModeBar": "hover",
        },
    )

    # -----------------------------------------------------
    # (B) 변화: 2019→2025 흐름(Play)
    # -----------------------------------------------------
    st.divider()
    st.subheader("권역이 만들어지는 7년의 흐름")

    # ✅ caption 대신 검은색 텍스트로
    st.markdown(
        "<div style='color:#111827; font-size:0.875rem; margin-top:0.15rem;'>"
        "▶︎ Play를 누르면, 자본이 어디로 모여 ‘굳어지는지’를 한 번에 볼 수 있습니다."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- state init
    if "kde_playing" not in st.session_state:
        st.session_state.kde_playing = False
    if "kde_year_cur" not in st.session_state:
        st.session_state.kde_year_cur = 2019
    if "kde_speed" not in st.session_state:
        st.session_state.kde_speed = 0.25

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
        # ✅ 재생 속도: 3단 고정 + 슬라이더 값 표시를 라벨로
        SPEED_VALUES = [0.5, 1.0, 2.0]  # (초)
        SPEED_LABELS = {0.5: "빠르게", 1.0: "보통", 2.0: "느리게"}

        # 현재 세션 값이 3단 중 하나가 아니면, 가장 가까운 값으로 스냅
        cur = float(st.session_state.kde_speed)
        if cur not in SPEED_VALUES:
            st.session_state.kde_speed = min(SPEED_VALUES, key=lambda v: abs(v - cur))

        speed_sec = st.slider(
            "재생 속도",
            min_value=min(SPEED_VALUES),
            max_value=max(SPEED_VALUES),
            value=float(st.session_state.kde_speed),
            step=None,  # 연속 슬라이더로 두되 format으로 '라벨'만 보이게 처리
            format="%s",
            key="kde_speed_streamlit",
        )

        # Streamlit slider가 반환하는 float를 3단으로 스냅(안정성)
        snapped = min(SPEED_VALUES, key=lambda v: abs(v - float(speed_sec)))
        st.session_state.kde_speed = float(snapped)

        # ✅ 값 표시는 숫자 대신 라벨로 보여주기 (format은 숫자를 못 바꾸는 케이스가 많아서 별도 표시)
        #    -> 슬라이더 아래에 라벨만 한 줄로 보여주면 UI가 깔끔해짐
        st.markdown(
            f"<div style='margin-top:-0.25rem; color:#EF4444; font-weight:600;'>"
            f"{SPEED_LABELS[float(st.session_state.kde_speed)]}"
            f"</div>",
            unsafe_allow_html=True,
        )


    # ✅ 연도 상태 문장 (차트 위)
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

        fig.update_layout(
            height=560,
            margin=dict(l=10, r=10, t=25, b=10),
        )
        return fig

    chart_slot.plotly_chart(
        make_single_year_fig(cur_year),
        width="stretch",
        config={
            "scrollZoom": True,
            "displayModeBar": "hover",
        },
    )

    # ---- play loop
    if st.session_state.kde_playing:
        time.sleep(float(st.session_state.kde_speed))

        if cur_year < 2025:
            st.session_state.kde_year_cur = cur_year + 1
        else:
            st.session_state.kde_playing = False

        st.rerun()



# =========================================================
# 6) 실행: [코드1] 다음에 [코드2]
# =========================================================
render_scatter_section()
st.divider()
render_kde_section()
