import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="보조배터리 노후화 예측",
    page_icon="🔋",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {background: #f5f7fb;}
    .main-title {font-size: 2.1rem; font-weight: 800; color: #172554;}
    .sub-title {color: #64748b; margin-bottom: 1.2rem;}
    .metric-card {
        background: white; padding: 18px; border-radius: 14px;
        border: 1px solid #e2e8f0; box-shadow: 0 2px 8px #0f172a0a;
    }
    .small-note {color: #64748b; font-size: 0.88rem;}
    div[data-testid="stAlert"] {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔋 보조배터리 노후화 예측 서비스</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">사용 조건을 바탕으로 배터리의 상대적인 건강 상태(SOH)를 추정합니다.</div>',
    unsafe_allow_html=True
)

st.info(
    "이 앱은 NASA 리튬이온 배터리 실험 데이터를 활용한 교육용 예측 예시입니다. "
    "실제 시중 보조배터리의 정확한 잔여 수명이나 화재 확률을 보장하지 않습니다."
)

# -------------------- 데이터 불러오기 --------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    candidates = [
        "nasa_battery_discharge_for_streamlit.csv",
        "battery_data.csv",
        "data.csv"
    ]
    for filename in candidates:
        try:
            return pd.read_csv(filename)
        except FileNotFoundError:
            pass
    return None

uploaded = st.sidebar.file_uploader("NASA 배터리 CSV 파일 업로드", type=["csv"])
data = load_data(uploaded)

# Column name normalizer: accepts the project's prepared CSV and common alternatives.
def find_column(columns, choices):
    lower_map = {str(c).strip().lower(): c for c in columns}
    for choice in choices:
        if choice.lower() in lower_map:
            return lower_map[choice.lower()]
    return None

if data is None:
    st.warning("데이터 CSV 파일을 찾지 못했습니다. 왼쪽 메뉴에서 CSV 파일을 업로드하거나 main.py와 같은 폴더에 넣어 주세요.")
    st.markdown("""
    **필요한 CSV 열 이름(권장 형식)**

    - `cycle`: 충·방전 사이클 횟수
    - `capacity_Ah`: 방전 용량(Ah)
    - `ambient_temperature_C`: 주변 온도(℃)
    - `battery_id`: 배터리 구분값(없어도 실행 가능)
    """)
    st.stop()

cycle_col = find_column(data.columns, ["cycle", "Cycle", "cycle_number"])
capacity_col = find_column(data.columns, ["capacity_Ah", "Capacity", "capacity", "discharge_capacity"])
temp_col = find_column(data.columns, ["ambient_temperature_C", "ambient_temperature", "Temperature", "temperature"])
battery_col = find_column(data.columns, ["battery_id", "Battery", "battery", "battery_name"])

if cycle_col is None or capacity_col is None:
    st.error("CSV에서 cycle과 capacity_Ah(또는 Capacity) 열을 찾지 못했습니다. CSV의 첫 행에 열 이름이 있는지 확인해 주세요.")
    st.write("현재 CSV 열 이름:", list(data.columns))
    st.stop()

work = pd.DataFrame()
work["cycle"] = pd.to_numeric(data[cycle_col], errors="coerce")
work["capacity"] = pd.to_numeric(data[capacity_col], errors="coerce")
work["temperature"] = (
    pd.to_numeric(data[temp_col], errors="coerce")
    if temp_col is not None else 25.0
)
if battery_col is not None:
    work["battery_id"] = data[battery_col].astype(str)
else:
    work["battery_id"] = "Battery 1"

work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["cycle", "capacity"])
work = work[work["capacity"] > 0]
if work.empty:
    st.error("분석할 수 있는 유효한 데이터가 없습니다. CSV의 cycle과 용량 값이 숫자인지 확인해 주세요.")
    st.stop()

# SOH = 현재 용량 / 배터리별 초기 용량 * 100
initial_capacity = work.groupby("battery_id")["capacity"].transform(
    lambda s: s.head(5).median() if len(s) else np.nan
)
work["soh"] = (work["capacity"] / initial_capacity) * 100
work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["soh"])
work["soh"] = work["soh"].clip(0, 120)

if len(work) < 5:
    st.error("예측 모델을 만들려면 유효한 데이터가 최소 5개 이상 필요합니다.")
    st.stop()

# -------------------- 모델 학습 --------------------
X = work[["cycle", "temperature"]].copy()
y = work["soh"].copy()

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X, y)
mae = mean_absolute_error(y, model.predict(X))

# -------------------- 사용자 입력 --------------------
st.sidebar.header("사용 조건 입력")
months = st.sidebar.slider("사용 기간(개월)", 1, 60, 12)
weekly_cycles = st.sidebar.slider("일주일 평균 충·방전 횟수", 0, 14, 4)
user_temp = st.sidebar.slider("주로 사용하는 주변 온도(℃)", 0, 50, 25)

st.sidebar.header("배터리 상태 확인")
swelling = st.sidebar.checkbox("배터리 또는 케이스가 부풀어 있음")
damage = st.sidebar.checkbox("케이스가 깨지거나 심하게 손상됨")
very_hot = st.sidebar.checkbox("사용·충전 중 비정상적으로 뜨거워짐")
often_hot = st.sidebar.checkbox("평소보다 자주 뜨거워짐")
smell = st.sidebar.checkbox("이상한 냄새, 연기 또는 소리가 남")
charging_problem = st.sidebar.checkbox("충전이 자주 끊기거나 작동이 불안정함")

estimated_cycles = months * 4.345 * weekly_cycles
estimated_cycles = max(1, estimated_cycles)

# Predict at the estimated usage cycle, using the user's temperature.
prediction = float(model.predict(pd.DataFrame({
    "cycle": [estimated_cycles],
    "temperature": [user_temp]
}))[0])
prediction = float(np.clip(prediction, 0, 120))

# Conservative display: the model's estimate is an educational approximation.
# Generate a curve from observed cycle range through the user's estimated cycle.
max_cycle = max(float(work["cycle"].max()), estimated_cycles)
cycle_grid = np.linspace(float(work["cycle"].min()), max_cycle, 100)
curve = model.predict(pd.DataFrame({
    "cycle": cycle_grid,
    "temperature": [user_temp] * len(cycle_grid)
}))
curve = np.minimum.accumulate(curve)
curve = np.clip(curve, 0, 120)

# -------------------- 결과 표시 --------------------
st.subheader("예측 결과")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("예상 누적 충·방전 횟수", f"{estimated_cycles:,.0f}회")
with c2:
    st.metric("예측 배터리 건강 상태(SOH)", f"{prediction:.1f}%")
with c3:
    status = "양호" if prediction >= 80 else "교체 검토"
    st.metric("모델 기준 상태", status)

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    number={"suffix": "%", "font": {"size": 34}},
    title={"text": "예측 SOH"},
    gauge={
        "axis": {"range": [0, 120]},
        "bar": {"color": "#2563eb"},
        "steps": [
            {"range": [0, 60], "color": "#fee2e2"},
            {"range": [60, 80], "color": "#fef3c7"},
            {"range": [80, 120], "color": "#dcfce7"}
        ],
        "threshold": {
            "line": {"color": "#dc2626", "width": 4},
            "thickness": 0.8,
            "value": 80
        }
    }
))
gauge.update_layout(height=280, margin=dict(l=20, r=20, t=55, b=15))
st.plotly_chart(gauge, use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("#### 충·방전 횟수에 따른 SOH 변화")
    fig = px.line(
        x=cycle_grid, y=curve,
        labels={"x": "누적 충·방전 횟수", "y": "예측 SOH(%)"},
        title="사용 온도 입력값을 적용한 예측 곡선"
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="SOH 80% 기준")
    fig.add_vline(x=estimated_cycles, line_dash="dot", line_color="green", annotation_text="입력 조건")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### 실험 데이터 분포")
    fig2 = px.scatter(
        work, x="cycle", y="soh", color="temperature",
        hover_data=["battery_id"],
        labels={"cycle": "충·방전 횟수", "soh": "SOH(%)", "temperature": "주변 온도(℃)"},
        title="NASA 실험 데이터의 배터리 건강 상태"
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

# Rough cycle estimate to 80% from predicted curve; no claim of real remaining life.
below = np.where(curve <= 80)[0]
if len(below) and prediction > 80:
    threshold_cycle = float(cycle_grid[below[0]])
    remaining_cycles = max(0, threshold_cycle - estimated_cycles)
    st.write(f"모델 곡선상 SOH 80%에 도달하는 누적 사이클은 약 **{threshold_cycle:,.0f}회**입니다.")
    st.caption(f"현재 입력값 이후 약 {remaining_cycles:,.0f}회의 추가 사이클로 계산되지만, 실험 데이터 기반의 단순 추정치입니다.")
elif prediction <= 80:
    st.warning("현재 입력 조건의 예측 SOH가 80% 이하입니다. 실제 제품 상태를 확인하고 교체를 검토하세요.")
else:
    st.caption("현재 모델 곡선에서는 SOH 80% 도달 지점을 계산할 수 없습니다.")

# -------------------- 안전 점검 --------------------
st.subheader("안전 상태 점검")
critical = swelling or damage or very_hot or smell
caution = often_hot or charging_problem or prediction < 80

if critical:
    st.error(
        "위험 징후가 선택되었습니다. 즉시 사용과 충전을 중단하고, 제품을 만지거나 분해하지 마세요. "
        "가연물에서 떨어진 안전한 장소에 두고 제조사 또는 적절한 안전 기관의 안내를 받으세요. "
        "연기·불꽃·급격한 발열이 있으면 가까이 가지 말고 119에 신고하세요."
    )
elif caution:
    st.warning(
        "주의가 필요합니다. 충전 중 자리를 비우지 말고, 이상 발열이나 충전 불량이 반복되면 사용을 중단한 뒤 "
        "제조사 점검 또는 교체를 고려하세요."
    )
else:
    st.success("선택한 항목 중 뚜렷한 위험 징후는 없습니다. 그래도 충전 중에는 제품 상태를 주기적으로 확인하세요.")

st.markdown("---")
st.markdown("### 분석의 한계")
st.write(
    "이 예측은 NASA의 실험용 리튬이온 셀 데이터를 학습한 교육용 모델입니다. "
    "실제 보조배터리는 셀 종류, 보호회로, 제조 품질, 사용 습관이 다르므로 결과가 실제 SOH와 다를 수 있습니다. "
    "SOH는 용량 기반 지표이며 부풀음·냄새·연기 같은 안전 문제를 예측하지 않습니다. "
    "안전 점검 결과는 사용자가 입력한 증상을 바탕으로 한 안내입니다."
)
st.caption(f"학습 데이터 행 수: {len(work):,} | 학습 데이터 기준 MAE: {mae:.2f}% (훈련 데이터 내 오차로, 일반화 성능 보장은 아님)")
