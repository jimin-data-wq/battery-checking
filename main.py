import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from datetime import date

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

/* 경고창은 밝은 배경 + 어두운 글씨로 고정해 가독성 확보 */
[data-testid="stAlert"] {
    background: #fff3cd !important;
    border: 1px solid #e7c66b !important;
    border-radius: 14px !important;
}
[data-testid="stAlert"] *,
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #49370a !important;
}
[data-testid="stAlert"] svg {
    color: #8a6500 !important;
    fill: #8a6500 !important;
}
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
        ["배터리 상태 확인", "NASA 데이터 분석", "LED 잔량 플래너", "사용 유형별 외출 예측"],
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
        # 상태 카드의 HTML은 하나의 markdown 블록 안에서 열고 닫아
        # 내용 없는 빈 초록색 칸이 따로 렌더링되는 현상을 방지합니다.
        st.markdown(
            f'''<div class="panel {state_class}" style="padding:18px 18px;border-radius:18px;">
                <div style="font-size:1.25rem;font-weight:800;line-height:1.45;color:#f5fbff;">{state_icon} {state}</div>
                <div style="font-size:1rem;line-height:1.65;margin-top:8px;color:#f5fbff;">{message}</div>
                <div style="font-size:.9rem;line-height:1.65;margin-top:9px;color:#e2edf8;">{advice}</div>
            </div>''',
            unsafe_allow_html=True
        )

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
elif page == "NASA 데이터 분석":
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


elif page == "LED 잔량 플래너":
    st.markdown('<div class="eyebrow">EVERYDAY POWER PLANNER</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">외출 전, 휴대폰 몇 번 충전할 수 있을까?</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">보조배터리의 불빛 칸 수만 선택하면 대략적인 충전 횟수를 알려드려요.</div>', unsafe_allow_html=True)

    st.markdown("### ① 보조배터리 불빛 칸 수")
    st.caption("현재 켜져 있는 LED 칸을 선택해 주세요. 4칸 표시등 기준의 대략적인 계산이에요.")
    if "led_level" not in st.session_state:
        st.session_state.led_level = 3

    led_cols = st.columns(4)
    for idx, col in enumerate(led_cols, start=1):
        with col:
            lit = idx <= st.session_state.led_level
            color = "#52e4ce" if lit else "#43536b"
            st.markdown(
                f"""<div style="background:rgba(22,39,66,.8);border:1px solid #60738f;border-radius:16px;padding:17px 5px;text-align:center;">
                <div style="width:22px;height:22px;border-radius:50%;background:{color};margin:0 auto 9px auto;box-shadow:0 0 15px {color if lit else 'transparent'};"></div>
                <div style="font-size:.82rem;color:#c8d8ef;">{idx}칸</div></div>""",
                unsafe_allow_html=True
            )
            if st.button(f"{idx}칸 선택", key=f"led_{idx}", use_container_width=True):
                st.session_state.led_level = idx
                st.rerun()

    level = st.session_state.led_level
    led_ratio = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}[level]
    approx_percent = int(led_ratio * 100)

    st.markdown("### ② 보조배터리 크기")
    size_choice = st.radio(
        "제품에 적힌 용량을 보고 가장 가까운 항목을 골라 주세요.",
        ["작은 편 · 10,000mAh", "일반적인 크기 · 20,000mAh", "큰 편 · 30,000mAh"],
        index=1,
        horizontal=True,
        label_visibility="collapsed"
    )
    powerbank_capacity = {
        "작은 편 · 10,000mAh": 10000,
        "일반적인 크기 · 20,000mAh": 20000,
        "큰 편 · 30,000mAh": 30000
    }[size_choice]

    # 일반적인 스마트폰(약 5,000mAh)과 평균적인 에너지 손실을 가정한 참고치
    estimated_charges = powerbank_capacity * led_ratio * 0.70 / 5000
    full_charge_count = int(estimated_charges)
    remainder = estimated_charges - full_charge_count

    st.markdown("### ③ 예상 충전 횟수")
    display_charges = max(0, round(estimated_charges))
    st.markdown(
        f"""<div class="panel" style="text-align:center;padding:24px 18px;margin-top:8px;">
        <div class="small-label">휴대폰 완충 기준 예상 횟수</div>
        <div style="font-size:2.8rem;font-weight:800;line-height:1.3;color:#52e4ce;">약 {display_charges}회</div>
        <div style="font-size:1rem;color:#e2edf8;margin-top:6px;">현재 LED {level}칸 · 잔량 약 {approx_percent}% 기준</div>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("### ④ 외출 시간에 맞춰 준비하기")
    st.caption("외출 시간과 현재 휴대폰 잔량을 선택하면 보조배터리를 더 충전해야 할지 알려드려요.")
    outing_hours = st.radio(
        "얼마나 외출할 예정인가요?",
        ["2시간", "5시간", "8시간"],
        index=1,
        horizontal=True,
        label_visibility="collapsed"
    )
    outing_hour_value = {"2시간": 2, "5시간": 5, "8시간": 8}[outing_hours]
    phone_battery = st.slider(
        "현재 휴대폰 배터리 잔량",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        format="%d%%"
    )

    # 평균적인 사용 상황을 가정한 단순 참고 계산: 시간당 약 10% 소모
    expected_phone_use = outing_hour_value * 10
    expected_phone_left = max(0, phone_battery - expected_phone_use)
    topup_needed = max(0, expected_phone_use - phone_battery)
    powerbank_available_percent = estimated_charges * 100

    if topup_needed == 0:
        st.success(
            f"현재 잔량으로도 {outing_hours} 외출을 버틸 가능성이 높아요. "
            f"일반적인 사용 기준으로 예상 잔량은 약 {expected_phone_left}%예요."
        )
    elif powerbank_available_percent >= topup_needed:
        st.success(
            f"{outing_hours} 외출 중 배터리가 부족해질 수 있지만, "
            "현재 보조배터리 잔량으로 보충할 수 있을 것으로 예상돼요."
        )
    else:
        st.warning(
            f"{outing_hours} 외출에는 보조배터리 잔량이 부족할 수 있어요. "
            "출발 전에 보조배터리를 더 충전하거나 충전 케이블을 챙기는 것을 추천해요."
        )

    with st.container(border=True):
        st.markdown("**계산 기준과 참고 사항**")
        st.caption("휴대폰 배터리가 평균적으로 시간당 약 10% 줄어든다고 가정한 간단한 참고 계산이에요. 게임·영상 촬영·내비게이션 사용이나 휴대폰 기종에 따라 실제 소모량은 크게 달라질 수 있어요. 보조배터리 충전 횟수 역시 제품과 변환 손실에 따라 달라집니다.")

    st.markdown("### 외출 전 간단 체크")
    a, b, c = st.columns(3)
    with a:
        st.checkbox("충전 케이블 챙기기", key="planner_cable")
    with b:
        st.checkbox("보조배터리 외관과 발열 확인", key="planner_safety")
    with c:
        st.checkbox("필요하면 출발 전 미리 충전", key="planner_topup")


else:
    st.markdown('<div class="eyebrow">SMART OUTING BATTERY FORECAST</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">내 사용 습관으로 외출 배터리 예측하기</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">어떤 앱을 얼마나 사용하는지 입력하면 화면 사용과 대기 시간을 나눠 배터리 소모량을 추정해요.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-soft">
      <b>사용 방법</b><br>
      <span class="muted">외출 시간, 출발 시 배터리 잔량, 앱별 예상 사용 시간을 입력하세요.
      앱마다 다른 시간당 소모율을 적용해 예상 잔량을 계산합니다.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ① 외출 조건")
    c1, c2, c3 = st.columns(3)
    with c1:
        outing_time = st.slider("총 외출 시간", min_value=1, max_value=12, value=5, step=1, format="%d시간")
    with c2:
        starting_battery = st.slider("출발할 때 휴대폰 잔량", min_value=5, max_value=100, value=80, step=5, format="%d%%")
    with c3:
        brightness = st.slider("평균 화면 밝기", min_value=10, max_value=100, value=60, step=5, format="%d%%")

    st.markdown("### ② 앱별 예상 사용 시간")
    st.caption("외출 시간 안에서 각 앱을 사용할 시간을 입력해 주세요. 사용하지 않을 앱은 0시간으로 두면 됩니다.")
    app_rates = {
        "메신저·문자": 4.0,
        "인터넷·검색": 7.0,
        "SNS·짧은 영상": 10.0,
        "동영상 시청": 13.0,
        "게임": 20.0,
        "지도·내비게이션": 16.0,
        "카메라·동영상 촬영": 18.0,
        "영상 통화": 17.0,
        "음악 감상": 5.0
    }
    app_hours = {}
    app_names = list(app_rates.keys())
    left_apps, right_apps = st.columns(2)
    for index, app_name in enumerate(app_names):
        target_col = left_apps if index % 2 == 0 else right_apps
        with target_col:
            app_hours[app_name] = st.number_input(
                f"{app_name} (시간)",
                min_value=0.0, max_value=float(outing_time), value=0.0, step=0.5,
                key=f"app_time_{index}"
            )

    active_hours = sum(app_hours.values())
    if active_hours > outing_time:
        st.error(f"앱 사용 시간 합계가 외출 시간({outing_time}시간)을 넘었어요. 현재 입력 합계는 {active_hours:.1f}시간입니다. 사용 시간을 조정해 주세요.")
        st.stop()

    idle_hours = outing_time - active_hours
    brightness_factor = 0.70 + (brightness / 100.0) * 0.60
    idle_rate = 1.5
    consumption_by_app = {
        name: hours * app_rates[name] * brightness_factor
        for name, hours in app_hours.items() if hours > 0
    }
    if idle_hours > 0:
        consumption_by_app["화면 꺼짐·대기"] = idle_hours * idle_rate

    total_consumption = sum(consumption_by_app.values())
    remaining_battery = max(0.0, starting_battery - total_consumption)
    estimated_raw = starting_battery - total_consumption

    st.markdown("### ③ 예상 배터리 변화")
    result_cols = st.columns([1.05, 1.4])
    with result_cols[0]:
        st.markdown(
            f"""<div class="panel" style="text-align:center;padding:22px 14px;">
            <div class="small-label">외출 종료 시 예상 잔량</div>
            <div style="font-size:2.65rem;font-weight:800;color:#52e4ce;line-height:1.25;">{remaining_battery:.0f}%</div>
            <div style="color:#dce9f8;font-size:.9rem;margin-top:8px;">출발 {starting_battery}% → 예상 {remaining_battery:.0f}%</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.metric("예상 소모량", f"{total_consumption:.1f}%")
        if estimated_raw < 0:
            st.warning("예측상 외출 도중 배터리가 모두 소모될 수 있어요. 보조배터리를 준비하는 것이 좋아요.")
        elif remaining_battery < 20:
            st.warning("외출 종료 전에 잔량이 20% 아래로 내려갈 수 있어요. 보조배터리를 챙기는 것을 추천해요.")
        elif remaining_battery < 40:
            st.info("배터리가 넉넉하지 않을 수 있어요. 화면 밝기를 낮추거나 사용 시간을 줄이면 도움이 됩니다.")
        else:
            st.success("입력한 사용 계획에서는 배터리가 비교적 여유로울 것으로 예상돼요.")

    with result_cols[1]:
        gauge_color = "#55dfc5" if remaining_battery >= 40 else "#ffc66d" if remaining_battery >= 20 else "#ff7f91"
        battery_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=remaining_battery,
            number={"suffix": "%", "font": {"size": 34, "color": "#f4f8ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#a9bddb"},
                "bar": {"color": gauge_color},
                "bgcolor": "rgba(255,255,255,.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "rgba(255,105,125,.20)"},
                    {"range": [20, 40], "color": "rgba(255,198,109,.18)"},
                    {"range": [40, 100], "color": "rgba(89,226,179,.14)"}
                ]
            }
        ))
        battery_gauge.update_layout(
            height=230, margin=dict(l=12, r=12, t=10, b=5),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#eaf2ff"
        )
        st.plotly_chart(battery_gauge, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### ④ 어떤 사용이 배터리를 많이 소모할까요?")
    if consumption_by_app:
        usage_fig = go.Figure(go.Bar(
            x=list(consumption_by_app.values()),
            y=list(consumption_by_app.keys()),
            orientation="h",
            marker=dict(
                color=list(consumption_by_app.values()),
                colorscale=[[0, "#3f9dba"], [0.55, "#55dfc5"], [1, "#ffc66d"]]
            ),
            text=[f"{v:.1f}%" for v in consumption_by_app.values()],
            textposition="outside"
        ))
        usage_fig.update_layout(
            height=max(260, 42 * len(consumption_by_app)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf2ff", xaxis_title="예상 배터리 소모 기여도(%p)",
            yaxis_title="", margin=dict(l=10, r=35, t=15, b=10),
            showlegend=False
        )
        st.plotly_chart(usage_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("앱 사용 시간을 입력하면 어떤 활동이 배터리를 많이 소모하는지 그래프로 보여드려요.")

    st.markdown("### ⑤ 배터리를 아끼는 맞춤 팁")
    if app_hours.get("게임", 0) > 0 or app_hours.get("동영상 시청", 0) > 0 or app_hours.get("영상 통화", 0) > 0:
        st.markdown('<div class="panel-soft"><b>화면·프로세서 사용 줄이기</b><br><span class="muted">게임, 영상, 영상 통화는 화면과 처리 장치를 오래 사용하므로 밝기를 낮추고 필요하지 않을 때 앱을 종료해 보세요.</span></div>', unsafe_allow_html=True)
    if app_hours.get("지도·내비게이션", 0) > 0:
        st.markdown('<div class="panel-soft"><b>내비게이션 사용 줄이기</b><br><span class="muted">화면을 계속 켜두지 않도록 음성 안내를 활용하고, 목적지 경로를 미리 확인하면 도움이 됩니다.</span></div>', unsafe_allow_html=True)
    if brightness >= 80:
        st.markdown('<div class="panel-soft"><b>화면 밝기 낮추기</b><br><span class="muted">평균 밝기가 높은 편이에요. 실내에서는 자동 밝기나 낮은 밝기를 활용해 보세요.</span></div>', unsafe_allow_html=True)
    if not any(app_hours.values()) and brightness < 80:
        st.markdown('<div class="panel-soft"><b>대기 중 전력 관리</b><br><span class="muted">사용하지 않는 앱의 백그라운드 새로고침과 불필요한 위치 사용을 줄이면 도움이 될 수 있어요.</span></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**예측 모델의 가정과 한계**")
        st.caption("앱별 시간당 소모율은 일반적인 사용 상황을 가정한 참고용 추정치이며 특정 스마트폰에서 직접 측정한 값이 아닙니다. 기기 성능, 통신 상태, 신호 세기, 배터리 노후도, 온도, 화면 주사율 등에 따라 실제 소모량은 달라집니다. 앱 사용 시간은 서로 겹치지 않게 입력해 주세요.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#8299bb;font-size:.8rem'>BATTERY CARE · NASA 실험 데이터 기반 교육용 프로젝트</div>",unsafe_allow_html=True)
