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


# =========================
# Plotly 렌더 유틸: 경계선 + 라벨(가독성 안정화: halo)
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

    # ✅ 히트맵(레드)과 의미 충돌 방지: 중립 회색 경계선
    return go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(width=1, color="#CBD5E1"),  # slate-300
        opacity=0.7,
        hoverinfo="skip",
        showlegend=False,
    )


def _label_colors_for_mode(b: gpd.GeoDataFrame, mode: str, year: int | None):
    if mode == "top_pct":
        black_set = {"용산구", "강남구", "서초구", "송파구"}
        colors = []
        for gu in b["gu"].tolist():
            c = "black" if gu in black_set else "white"
            if year is not None and (int(year) == 2019 and gu == "강남구"):
                c = "white"
            colors.append(c)
        return colors
    return ["black"] * len(b)


def make_label_traces_with_halo(b3857: gpd.GeoDataFrame, mode: str, year: int | None):
    """
    Plotly는 text stroke(외곽선)가 직접 지원이 약해서
    ✅ 동일 텍스트를 2개 trace로 겹쳐 halo 효과를 만든다.
      - 아래(halo): 검정/흰색(대비용), 큰 글자
      - 위(본문): 원래 색, 작은 글자
    """
    if "cx" not in b3857.columns or "cy" not in b3857.columns:
        b = b3857.copy()
        b["cx"] = b.geometry.centroid.x
        b["cy"] = b.geometry.centroid.y
    else:
        b = b3857

    main_colors = _label_colors_for_mode(b, mode, year)

    # halo 색: 본문이 white면 black halo, 본문이 black이면 white halo
    halo_colors = [("black" if c == "white" else "white") for c in main_colors]

    halo = go.Scatter(
        x=b["cx"],
        y=b["cy"],
        mode="text",
        text=b["gu"],
        textposition="middle center",
        textfont=dict(size=14, color=halo_colors),  # halo: 조금 크게
        hoverinfo="skip",
        showlegend=False,
    )

    main = go.Scatter(
        x=b["cx"],
        y=b["cy"],
        mode="text",
        text=b["gu"],
        textposition="middle center",
        textfont=dict(size=11, color=main_colors),  # main: 원래 크기
        hoverinfo="skip",
        showlegend=False,
    )

    return halo, main


def heatmap_trace(Z, extent, vmax, colorscale, show_scale, colorbar_x=1.02):
    if Z is None:
        return None

    xmin, xmax, ymin, ymax = extent
    ny, nx = Z.shape
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    # ✅ 범례(컬러바) 더 키우기 + 타이틀/틱 가독성 상승
    #    plotly 호환: title은 dict로( titlefont 금지 )
    cb = dict(
        thickness=18,
        len=0.68,
        x=colorbar_x,
        y=0.5,
        tickfont=dict(size=11, color="#4B5563"),
        title=dict(text="Value-weighted KDE", font=dict(size=12, color="#374151")),
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
        hovertemplate="kde=%{z:.3g}<extra></extra>",
    )


# =========================================================
# 5) KDE 섹션 (Plotly 렌더링)
# =========================================================
def render_kde_section():
    st.header("Seoul KDE")

    DEFAULT_CAP = 9000
    DEFAULT_BW = 0.22
    DEFAULT_GRIDSIZE = 250

    path_tx = CSV_PATH_DEFAULT
    geojson_url = GEOJSON_URL_DEFAULT

    show_options = st.toggle("옵션", value=False, key="kde_show_options")

    if show_options:
        row1a, row1b = st.columns([1.4, 1.4], vertical_alignment="center")

        with row1a:
            mode_ui = st.radio("모드", ["전체 거래", "상위 10%"], index=1, horizontal=True, key="mode_ui")
            mode = "raw" if mode_ui == "전체 거래" else "top_pct"

        with row1b:
            value_label = st.radio("값 컬럼", list(VALUE_LABEL_TO_COL.keys()), index=1, horizontal=True, key="value_label_ui")
            value_col = VALUE_LABEL_TO_COL[value_label]

        row2a, row2b, row2c = st.columns([1.4, 1.4, 1.4], vertical_alignment="center")
        with row2a:
            cap = st.slider("cap(합성점 수)", 1000, 20000, int(DEFAULT_CAP), 500, key="cap_ui")
        with row2b:
            bw = st.slider("bw(bandwidth)", 0.05, 1.00, float(DEFAULT_BW), 0.01, key="bw_ui")
        with row2c:
            gridsize = st.slider("gridsize", 120, 400, int(DEFAULT_GRIDSIZE), 10, key="gridsize_ui")

        st.divider()
    else:
        mode_ui = st.session_state.get("mode_ui", "상위 10%")
        mode = "raw" if mode_ui == "전체 거래" else "top_pct"

        value_label = st.session_state.get("value_label_ui", "평당 거래가격")
        if value_label not in VALUE_LABEL_TO_COL:
            value_label = "평당 거래가격"
        value_col = VALUE_LABEL_TO_COL[value_label]

        cap = int(st.session_state.get("cap_ui", DEFAULT_CAP))
        bw = float(st.session_state.get("bw_ui", DEFAULT_BW))
        gridsize = int(st.session_state.get("gridsize_ui", DEFAULT_GRIDSIZE))

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
    top_pct = 0.10
    colorscale = "RdYlBu_r"  # plotly built-in

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

    xmin, xmax, ymin, ymax = extent

    # -----------------------------------------------------
    # (A) 거래 금액 가중 KDE 비교
    #  - 모드바는 hover에서만 표시(시각적 잡음 ↓)
    #  - 안내 캡션 1줄 추가(줌 동기화 힌트)
    # -----------------------------------------------------
    st.header("거래 금액 가중 KDE 비교")
    st.caption(make_condition_text_simple(mode=mode, value_label=value_label, bw=float(bw)))
    st.caption("두 지도는 줌/이동이 동기화됩니다. 한쪽만 조작해도 함께 움직여요.")

    colL, colR = st.columns([1, 1], vertical_alignment="top")
    with colL:
        year_left = st.selectbox("연도(좌)", years_all, index=0, key="year_left_compare")
    with colR:
        year_right = st.selectbox("연도(우)", years_all, index=len(years_all) - 1, key="year_right_compare")

    ZL = Z_by_year.get(int(year_left))
    ZR = Z_by_year.get(int(year_right))

    fig_cmp = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.03,
        subplot_titles=(f"{int(year_left)}년", f"{int(year_right)}년"),
    )

    hmL = heatmap_trace(ZL, extent, vmax_fixed, colorscale, show_scale=False)
    hmR = heatmap_trace(ZR, extent, vmax_fixed, colorscale, show_scale=True, colorbar_x=1.03)

    if hmL is not None:
        fig_cmp.add_trace(hmL, row=1, col=1)
    else:
        fig_cmp.add_annotation(text=f"{int(year_left)}년<br>데이터 없음", x=0.22, y=0.5, xref="paper", yref="paper", showarrow=False)

    if hmR is not None:
        fig_cmp.add_trace(hmR, row=1, col=2)
    else:
        fig_cmp.add_annotation(text=f"{int(year_right)}년<br>데이터 없음", x=0.78, y=0.5, xref="paper", yref="paper", showarrow=False)

    bd = make_boundary_traces(b3857)
    haloL, mainL = make_label_traces_with_halo(b3857, mode=mode, year=int(year_left))
    haloR, mainR = make_label_traces_with_halo(b3857, mode=mode, year=int(year_right))

    fig_cmp.add_trace(bd, row=1, col=1)
    fig_cmp.add_trace(bd, row=1, col=2)
    fig_cmp.add_trace(haloL, row=1, col=1)
    fig_cmp.add_trace(mainL, row=1, col=1)
    fig_cmp.add_trace(haloR, row=1, col=2)
    fig_cmp.add_trace(mainR, row=1, col=2)

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
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig_cmp,
        width="stretch",
        config={
            "scrollZoom": True,
            "displayModeBar": "hover",  # ✅ 항상 표시 X
        },
    )

    st.divider()

    # -----------------------------------------------------
    # (B) 거래 금액 가중 KDE의 변화
    #  - Streamlit 컨트롤(Play/연도/속도) 유지
    #  - Plotly 내부 슬라이더/버튼 제거(차트 위 슬라이더 제거)
    #  - 모드바 hover로 변경
    # -----------------------------------------------------
    st.header("거래 금액 가중 KDE의 변화")
    st.caption(make_condition_text_simple(mode=mode, value_label=value_label, bw=float(bw)))

    # ---- state init
    if "kde_playing" not in st.session_state:
        st.session_state.kde_playing = False
    if "kde_year_cur" not in st.session_state:
        st.session_state.kde_year_cur = 2019
    if "kde_speed" not in st.session_state:
        st.session_state.kde_speed = 0.25

    # ---- controls (예전처럼: Play / 연도 / 속도)
    ctrl1, ctrl2, ctrl3 = st.columns([1.0, 2.2, 2.2], vertical_alignment="center")

    with ctrl1:
        c1a, c1b = st.columns(2)
        with c1a:
            if st.button("▶ Play", use_container_width=True, key="kde_btn_play"):
                st.session_state.kde_playing = True
        with c1b:
            if st.button("■ Pause", use_container_width=True, key="kde_btn_pause"):
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
        speed_sec = st.slider(
            "재생 속도(초)",
            min_value=0.05,
            max_value=1.00,
            step=0.05,
            value=float(st.session_state.kde_speed),
            key="kde_speed_streamlit",
        )
        st.session_state.kde_speed = float(speed_sec)

    chart_slot = st.empty()

    def make_single_year_fig(year: int):
        Zy = Z_by_year.get(int(year))

        fig = go.Figure()

        hm = heatmap_trace(Zy, extent, vmax_fixed, colorscale, show_scale=True, colorbar_x=1.03)
        if hm is not None:
            fig.add_trace(hm)
        else:
            fig.add_annotation(text=f"{int(year)}년<br>데이터 없음", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)

        fig.add_trace(make_boundary_traces(b3857))

        halo, main = make_label_traces_with_halo(b3857, mode=mode, year=int(year))
        fig.add_trace(halo)
        fig.add_trace(main)

        fig.update_xaxes(range=[xmin, xmax], visible=False, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[ymin, ymax], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)

        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=45, b=10),
            title=dict(text=f"{int(year)}년", x=0.5),
        )
        return fig

    # ---- render current
    chart_slot.plotly_chart(
        make_single_year_fig(int(st.session_state.kde_year_cur)),
        width="stretch",
        config={
            "scrollZoom": True,
            "displayModeBar": "hover",  # ✅ 항상 표시 X
        },
    )

    # ---- play loop (Streamlit rerun)
    if st.session_state.kde_playing:
        time.sleep(float(st.session_state.kde_speed))

        cur = int(st.session_state.kde_year_cur)
        if cur < 2025:
            st.session_state.kde_year_cur = cur + 1
        else:
            st.session_state.kde_playing = False

        st.rerun()


# =========================================================
# 6) 실행: [코드1] 다음에 [코드2]
# =========================================================
render_scatter_section()
st.divider()
render_kde_section()
