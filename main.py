import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Battery Care | 보조배터리 건강 진단",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "nasa_battery_discharge_for_streamlit.csv"
IMAGE_PATH = BASE_DIR / "battery_3d.png"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Noto Sans KR',sans-serif;}
.stMarkdown, .stMarkdown p, .stMarkdown li, label, [data-testid="stMetricLabel"] {
    word-break: keep-all;
    overflow-wrap: anywhere;
    line-height: 1.6;
}
.stMarkdown p {margin-bottom: .55rem;}
[data-testid="column"] {min-width: 0 !important;}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label {
    white-space: normal !important;
    line-height: 1.45 !important;
}
[data-testid="stMetricValue"] {font-size: clamp(1.15rem, 2vw, 1.8rem) !important;}
[data-testid="stMetricLabel"] {font-size: .88rem !important; white-space: normal !important;}
.hero-title {font-size: clamp(1.55rem, 3vw, 2.2rem); line-height: 1.3; word-break: keep-all;}
.hero-sub {line-height: 1.65; word-break: keep-all;}
.panel, .panel-soft {min-width: 0; overflow-wrap: anywhere;}\ndiv[data-testid="stVerticalBlock"] {min-width: 0;}\n[data-testid="stMarkdownContainer"] {min-width: 0;}
.stApp {
    background: radial-gradient(ellipse at 50% 0%, #16294b 0%, #0b1427 48%, #080f1e 100%);
    color: #edf5ff;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#111f39 0%,#0c172b 100%);
    border-right: 1px solid rgba(135,180,255,.16);
}
section[data-testid="stSidebar"] * {color:#e8f2ff;}
.block-container {padding-top:1.6rem; padding-bottom:2.5rem; max-width:1400px;}
h1,h2,h3,h4,p,label {color:#edf5ff;}
.hero-title {font-size:2.2rem;font-weight:800;letter-spacing:-.04em;margin:0;}
.hero-sub {color:#a9bddb;font-size:1rem;margin-top:.3rem;margin-bottom:1.1rem;}
.eyebrow {color:#63e6d3;font-weight:700;letter-spacing:.12em;font-size:.78rem;}
.panel {
    background:linear-gradient(145deg,rgba(28,47,79,.88),rgba(16,28,51,.94));
    border:1px solid rgba(142,181,235,.22);
    border-radius:22px;padding:20px 22px;
    box-shadow:0 14px 35px rgba(0,0,0,.18);
}
.panel-soft {
    background:rgba(19,34,60,.72);border:1px solid rgba(142,181,235,.18);
    border-radius:18px;padding:16px 18px;
}
.status-good {background:linear-gradient(120deg,rgba(24,111,91,.95),rgba(18,65,73,.95));border:1px solid rgba(106,242,194,.6);}
.status-warn {background:linear-gradient(120deg,rgba(132,91,31,.95),rgba(75,56,35,.95));border:1px solid rgba(255,202,111,.6);}
.status-bad {background:linear-gradient(120deg,rgba(133,48,61,.96),rgba(74,31,50,.96));border:1px solid rgba(255,133,147,.65);}
.big-number {font-size:2.5rem;font-weight:800;line-height:1.1;color:#f5fbff;}
.muted {color:#a9bddb;font-size:.9rem;}
.small-label {color:#9fb5d6;font-size:.82rem;font-weight:600;}
.tip {border-left:3px solid #53ddca;padding:8px 0 8px 13px;color:#d8e8fb;}
div[data-testid="stMetric"] {
    background:rgba(12,24,45,.55);padding:14px 16px;border-radius:15px;
    border:1px solid rgba(142,181,235,.18);
}
div[data-testid="stMetricLabel"] {color:#a9bddb;}
div[data-testid="stMetricValue"] {color:#f5fbff;}
.stButton>button {
    background:linear-gradient(90deg,#45d9d0,#8df0b4);color:#082033;
    border:0;border-radius:14px;font-weight:800;min-height:2.8rem;
}
hr {border-color:rgba(142,181,235,.18);}
[data-testid="stDataFrame"] {background:#10203a;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 데이터와 모델
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    if not CSV_PATH.exists():
        return None
    df = pd.read_csv(CSV_PATH)
    required = {"battery_id", "cycle", "ambient_temperature_C", "capacity_Ah"}
    if not required.issubset(df.columns):
        return None
    for col in ["cycle", "ambient_temperature_C", "capacity_Ah"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["battery_id", "cycle", "ambient_temperature_C", "capacity_Ah"])
    df = df[df["capacity_Ah"] > 0].copy()
    df = df.sort_values(["battery_id", "cycle"])
    # 각 배터리의 초반 5회 방전 용량 중앙값을 초기 용량으로 설정
    first_capacity = df.groupby("battery_id")["capacity_Ah"].transform(
        lambda s: s.head(5).median()
    )
    df["SOH"] = (df["capacity_Ah"] / first_capacity) * 100
    df["SOH"] = df["SOH"].clip(0, 120)
    return df

@st.cache_resource
def train_model(data):
    features = data[["cycle", "ambient_temperature_C"]]
    target = data["SOH"]
    model = RandomForestRegressor(
        n_estimators=240, max_depth=12, min_samples_leaf=3,
        random_state=42
    )
    model.fit(features, target)
    pred = model.predict(features)
    mae = mean_absolute_error(target, pred)
    return model, mae

data = load_dataset()
if data is None:
    st.error("데이터 CSV를 찾지 못했어요. main.py와 같은 폴더에 nasa_battery_discharge_for_streamlit.csv 파일을 넣어 주세요.")
    st.stop()

model, train_mae = train_model(data)

# ---------------------------------------------------------
# 공통 헤더와 메뉴
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔋 Battery Care")
    st.caption("보조배터리 건강 상태 도우미")
    page = st.radio(
        "페이지 선택",
        ["배터리 상태 확인", "NASA 데이터 분석"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("NASA 실험 데이터를 바탕으로 배터리 상태를 참고용으로 분석합니다.")
    st.caption("※ 교육·탐구용 서비스이며 실제 제품의 안전성을 보증하지 않습니다.")

st.markdown('<div class="eyebrow">SMART BATTERY HEALTH CHECK</div>', unsafe_allow_html=True)
if page == "배터리 상태 확인":
    st.markdown('<div class="hero-title">내 보조배터리, 지금 건강할까요?</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">사용 정보를 입력하면 배터리 건강 상태를 쉽게 확인할 수 있어요.</div>', unsafe_allow_html=True)

    # Inputs
    with st.sidebar:
        st.markdown("### ⚙️ 내 사용 정보")
        months = st.slider("사용 기간", 1, 60, 12, help="보조배터리를 사용한 개월 수")
        weekly_cycles = st.slider("일주일 평균 충·방전 횟수", 0, 14, 4)
        ambient_temp = st.slider("주로 사용하는 주변 온도", 0, 45, 25, format="%d ℃")
        st.markdown("---")
        st.markdown("### 🛡️ 외관·사용 중 이상 징후")
        swollen = st.checkbox("배터리나 케이스가 부풀어 올랐어요")
        damaged = st.checkbox("케이스가 깨지거나 심하게 손상됐어요")
        abnormal_heat = st.checkbox("충전·사용 중 비정상적으로 뜨거워져요")
        smell_smoke = st.checkbox("이상한 냄새·연기·소리가 나요")
        frequent_heat = st.checkbox("평소보다 발열이 자주 발생해요")
        charge_issue = st.checkbox("충전이 자주 끊기거나 작동이 불안정해요")

    total_cycles = max(1, months * 4.345 * weekly_cycles)
    # User-entered cycle count may exceed the lab range; flag as extrapolation.
    cycle_min = float(data["cycle"].min())
    cycle_max = float(data["cycle"].max())
    model_cycle = float(np.clip(total_cycles, cycle_min, cycle_max))
    predicted_soh = float(model.predict(pd.DataFrame({
        "cycle": [model_cycle],
        "ambient_temperature_C": [ambient_temp]
    }))[0])
    predicted_soh = float(np.clip(predicted_soh, 0, 120))

    critical = swollen or damaged or abnormal_heat or smell_smoke
    caution = frequent_heat or charge_issue or predicted_soh < 80

    if critical:
        state, state_class, state_icon = "즉시 사용 중단", "status-bad", "🚨"
        message = "배터리 안전에 위험할 수 있는 징후가 있어요."
        advice = "전원과 충전을 즉시 중단하고 제품을 분해하거나 누르지 마세요. 연기·불꽃·급격한 발열이 있으면 가까이 가지 말고 119에 신고하세요."
        risk = "높음"
    elif caution:
        state, state_class, state_icon = "주의가 필요해요", "status-warn", "⚠️"
        message = "평소보다 세심하게 상태를 확인해 주세요."
        advice = "발열과 충전 상태를 관찰하고, 이상이 반복되면 사용을 중단한 뒤 제조사 점검이나 교체를 고려하세요."
        risk = "주의"
    else:
        state, state_class, state_icon = "현재 상태 양호", "status-good", "✓"
        message = "입력한 정보에서 뚜렷한 위험 징후는 발견되지 않았어요."
        advice = "고온·직사광선을 피하고, 충전 중에는 제품 상태를 가끔 확인해 주세요."
        risk = "낮음"

    left, center, right = st.columns([0.92, 1.55, 0.92], gap="medium")
    with left:
        st.markdown("#### 📋 사용 정보")
        st.markdown(f"**사용 기간**  \n{months}개월")
        st.markdown(f"**예상 누적 충·방전**  \n{total_cycles:,.0f}회")
        st.markdown(f"**주변 온도**  \n{ambient_temp} ℃")
        st.markdown(f"**안전 위험도**  \n{risk}")

    with center:
        if IMAGE_PATH.exists():
            st.image(str(IMAGE_PATH), use_container_width=True)
        else:
            st.markdown("<div style='font-size:7rem;text-align:center;padding:2rem;'>🔋</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="panel {state_class}" style="padding:18px 14px;border-radius:18px;">', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.25rem;font-weight:800'>{state_icon} {state}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1rem;margin-top:7px'>{message}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='muted' style='margin-top:9px'>{advice}</div>", unsafe_allow_html=True)

    with right:
        st.markdown("#### 배터리 건강 점수")
        gauge_color = "#59e2b3" if predicted_soh >= 80 else ("#ffc66d" if predicted_soh >= 60 else "#ff7787")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=predicted_soh,
            number={"suffix":"%", "font":{"size":34,"color":"#f4f8ff"}},
            gauge={
                "axis":{"range":[0,120],"tickcolor":"#a9bddb"},
                "bar":{"color":gauge_color},
                "bgcolor":"rgba(255,255,255,.05)",
                "borderwidth":0,
                "steps":[
                    {"range":[0,60],"color":"rgba(255,105,125,.18)"},
                    {"range":[60,80],"color":"rgba(255,198,109,.18)"},
                    {"range":[80,120],"color":"rgba(89,226,179,.15)"}
                ],
                "threshold":{"line":{"color":"#ffffff","width":3},"thickness":.8,"value":80}
            }
        ))
        gauge.update_layout(height=220, margin=dict(l=12,r=12,t=8,b=8), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar":False})
        st.markdown("<div class='muted'>SOH = 처음 용량 대비 현재 용량의 비율</div>", unsafe_allow_html=True)

    st.markdown("")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("예상 누적 충·방전 횟수", f"{total_cycles:,.0f}회")
    metric2.metric("예측 SOH", f"{predicted_soh:.1f}%")
    metric3.metric("안전 징후 입력", "있음" if critical or caution else "없음")

    # Simple trend chart
    st.markdown("#### 📉 사용 횟수에 따른 건강 상태 예상")
    st.caption("현재 사용 온도를 가정했을 때의 모델 예측 흐름입니다. 실험 데이터의 범위를 넘어서는 경우 실제 결과와 차이가 클 수 있어요.")
    max_x = max(cycle_max, model_cycle)
    grid = np.linspace(cycle_min, cycle_max, 80)
    vals = model.predict(pd.DataFrame({
        "cycle": grid,
        "ambient_temperature_C": [ambient_temp] * len(grid)
    }))
    vals = np.minimum.accumulate(vals)
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=grid,y=vals,mode="lines",name="예측 SOH",
                               line=dict(color="#5ee6d1",width=3),
                               fill="tozeroy",fillcolor="rgba(94,230,209,.10)"))
    chart.add_hline(y=80,line_dash="dash",line_color="#ffc66d",annotation_text="80% 참고선")
    if total_cycles <= cycle_max:
        chart.add_vline(x=total_cycles,line_dash="dot",line_color="#ffffff",annotation_text="내 사용 조건")
    chart.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",font_color="#eaf2ff",
                        xaxis_title="누적 충·방전 횟수(회)",yaxis_title="배터리 건강 상태(SOH, %)",
                        margin=dict(l=10,r=10,t=30,b=10),showlegend=False)
    st.plotly_chart(chart,use_container_width=True,config={"displayModeBar":False})
    if total_cycles > cycle_max:
        st.warning(f"입력한 예상 누적 횟수({total_cycles:,.0f}회)가 NASA 실험 데이터의 최대 사이클({cycle_max:,.0f}회)을 넘어섰어요. 따라서 현재 점수는 실험 범위의 끝값으로 계산한 참고치이며, 정확한 외삽 예측이 아닙니다.")

    st.markdown("#### 💡 지금 실천하면 좋은 관리 방법")
    tips = [
        "뜨거운 차 안이나 직사광선 아래에 보조배터리를 오래 두지 않기",
        "충전 중 심한 발열·냄새·부풀음이 생기면 즉시 사용을 중단하기",
        "케이블과 충전 단자에 손상이나 이물질이 없는지 확인하기"
    ]
    t1,t2,t3 = st.columns(3)
    for col,tip in zip([t1,t2,t3],tips):
        with col:
            st.markdown(f'<div class="panel-soft"><div class="tip">✓ {tip}</div></div>',unsafe_allow_html=True)

    st.caption("중요: SOH는 배터리 용량을 기준으로 한 건강 지표입니다. 부풀음·연기·냄새 같은 안전 문제를 예측하는 값이 아닙니다. 이 앱은 교육·탐구용 참고 도구이며 실제 제품의 안전 판정이나 잔여 수명을 보장하지 않습니다.")

# ---------------------------------------------------------
# 두 번째 페이지: NASA 데이터 설명
# ---------------------------------------------------------
else:
    st.markdown('<div class="hero-title">NASA 데이터로 배터리 노화를 알아봐요</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">어려운 실험 데이터를 쉽게 풀어서, 앱이 어떤 근거로 예측하는지 설명합니다.</div>', unsafe_allow_html=True)

    st.markdown("### 🧪 NASA는 어떤 실험을 했을까요?")
    st.write(
        "NASA 연구진은 리튬이온 배터리 셀을 여러 번 충전하고 방전하면서 성능이 어떻게 달라지는지 관찰했어요. "
        "반복 횟수, 주변 온도, 방전할 때 측정한 용량 등을 기록해 배터리가 시간이 지나며 얼마나 약해지는지 살펴본 데이터입니다."
    )

    a,b,c = st.columns(3)
    a.metric("데이터 기록 수", f"{len(data):,}개")
    b.metric("실험 배터리 수", f"{data['battery_id'].nunique()}개")
    c.metric("최대 기록 사이클", f"{data['cycle'].max():,.0f}회")

    st.markdown("### 📚 데이터 속성, 쉽게 이해하기")
    explanation = pd.DataFrame([
        ["battery_id", "배터리 구분 이름", "어떤 배터리 셀에서 나온 기록인지 구분해요."],
        ["cycle", "충·방전 반복 횟수", "배터리를 몇 번 사용했는지 나타내는 숫자예요."],
        ["ambient_temperature_C", "주변 온도(℃)", "실험을 진행한 장소의 온도예요."],
        ["capacity_Ah", "방전 용량(Ah)", "충전된 전기를 실제로 얼마나 꺼내 쓸 수 있었는지 보여줘요."],
        ["mean_temperature_C", "평균 배터리 온도(℃)", "방전 실험 중 배터리에서 측정된 평균 온도예요."],
        ["max_temperature_C", "최고 배터리 온도(℃)", "실험 중 배터리에서 측정된 가장 높은 온도예요."],
        ["mean_voltage_V", "평균 전압(V)", "방전하는 동안의 평균 전압이에요."],
        ["mean_current_A", "평균 전류(A)", "방전하는 동안 전류가 어느 정도 흘렀는지 나타내요."],
        ["discharge_time_s", "방전 시간(초)", "방전이 진행된 시간이에요."],
        ["experiment_time", "실험 시점", "실험 기록의 시간 정보예요."]
    ], columns=["데이터 열 이름","뜻","쉽게 설명하면"])
    st.dataframe(explanation,use_container_width=True,hide_index=True)

    st.markdown("### 🔋 SOH는 어떻게 계산했을까요?")
    st.write("SOH는 State of Health의 약자로, 배터리가 처음과 비교해 얼마나 많은 용량을 유지하고 있는지 나타내는 지표예요.")
    st.latex(r"SOH(\%) = \frac{현재\ 방전\ 용량}{초기\ 방전\ 용량} \times 100")
    st.write("예를 들어 처음 용량이 2.0Ah였는데 지금 1.6Ah만 저장·방전할 수 있다면, SOH는 80%예요. 이 앱에서는 각 배터리의 처음 5회 기록 용량 중앙값을 초기 용량으로 사용했어요.")

    st.markdown("### 📉 실험에서는 어떤 변화가 보일까요?")
    by_cycle = data.copy()
    by_cycle["cycle_group"] = (by_cycle["cycle"] // 25 * 25).astype(int)
    trend = by_cycle.groupby("cycle_group",as_index=False).agg(
        mean_soh=("SOH","mean"), min_soh=("SOH","min"), max_soh=("SOH","max")
    )
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=trend["cycle_group"],y=trend["mean_soh"],
                                   mode="lines+markers",name="평균 SOH",
                                   line=dict(color="#5ee6d1",width=3)))
    trend_fig.add_trace(go.Scatter(x=trend["cycle_group"],y=trend["max_soh"],
                                   mode="lines",name="최고 SOH",
                                   line=dict(color="#8ca9ff",dash="dot")))
    trend_fig.add_trace(go.Scatter(x=trend["cycle_group"],y=trend["min_soh"],
                                   mode="lines",name="최저 SOH",
                                   line=dict(color="#ffc66d",dash="dot"),
                                   fill="tonexty",fillcolor="rgba(140,169,255,.08)"))
    trend_fig.update_layout(height=360,paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",font_color="#eaf2ff",
                            xaxis_title="충·방전 횟수(회)",yaxis_title="SOH(%)",
                            margin=dict(l=10,r=10,t=20,b=10),hovermode="x unified")
    st.plotly_chart(trend_fig,use_container_width=True,config={"displayModeBar":False})
    st.info("그래프의 선은 여러 배터리 실험 기록을 사이클 구간별로 묶어 평균과 범위를 나타낸 것입니다. 모든 배터리가 똑같은 속도로 노화한다는 뜻은 아니에요.")

    st.markdown("### 🤖 앱의 예측은 어떤 순서로 이뤄질까요?")
    steps = [
        ("1. 실험 기록 읽기","NASA 데이터에서 충·방전 횟수, 주변 온도, 방전 용량을 가져와요."),
        ("2. 초기 용량과 비교하기","각 배터리의 초반 용량을 기준으로 SOH를 계산해요."),
        ("3. 관계 학습하기","랜덤 포레스트 모델이 사이클 횟수와 주변 온도에 따라 SOH가 어떻게 달라지는지 학습해요."),
        ("4. 사용 조건에 적용하기","첫 페이지에서 입력한 예상 사용 횟수와 온도를 모델에 넣어 SOH를 추정해요.")
    ]
    for title,desc in steps:
        st.markdown(f'<div class="panel-soft" style="margin-bottom:10px"><b>{title}</b><br><span class="muted">{desc}</span></div>',unsafe_allow_html=True)

    st.markdown("### 🔎 이 분석으로 알 수 있는 것과 알 수 없는 것")
    yes,no = st.columns(2)
    with yes:
        with st.container(border=True):
            st.markdown("#### ✅ 알 수 있는 것")
            st.markdown(
                "- 반복 사용에 따라 방전 용량이 어떻게 달라지는지  \n"
                "- 실험 데이터에서 온도와 SOH 사이에 어떤 관계가 나타나는지  \n"
                "- 입력 조건에서 모델이 추정한 참고용 SOH"
            )
    with no:
        with st.container(border=True):
            st.markdown("#### ⚠️ 이 데이터만으로 알 수 없는 것")
            st.markdown(
                "- 시중 보조배터리 제품 각각의 정확한 실제 수명  \n"
                "- 부풀음이나 화재가 발생할 확률  \n"
                "- 모든 배터리 종류에 동일하게 적용되는 안전 기준"
            )

    st.markdown("### 📌 탐구 보고서에 활용할 핵심 해석")
    st.write(
        "NASA 실험 데이터는 리튬이온 배터리가 충·방전을 반복하면서 방전 용량이 감소하는 경향을 분석하는 데 활용할 수 있어요. "
        "이 앱은 초기 용량 대비 현재 용량을 SOH로 바꾸고, 충·방전 횟수와 주변 온도를 입력값으로 사용해 SOH를 예측합니다. "
        "다만 실험용 셀과 실제 보조배터리는 구조와 보호회로, 사용 환경이 다르므로 결과는 실제 제품의 수명을 확정하는 값이 아니라 경향을 이해하기 위한 참고 자료로 해석해야 합니다."
    )
    st.caption(f"참고: 학습 데이터 전체에서 계산한 평균 절대 오차(MAE)는 {train_mae:.2f}%입니다. 이는 학습에 사용한 데이터에서의 오차이므로, 새로운 제품에 대한 예측 정확도를 뜻하지 않습니다.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#8299bb;font-size:.8rem'>BATTERY CARE · NASA 실험 데이터 기반 교육용 프로젝트</div>",unsafe_allow_html=True)
