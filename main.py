footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div style="font-size: .9rem; letter-spacing:.12em; font-weight:700; color:#9ee5d4;">BATTERY CARE · PERSONAL CHECK</div>
  <h1>내 보조배터리, 계속 써도 괜찮을까?</h1>
  <p>몇 가지 쉬운 정보를 입력하면 NASA 리튬이온 배터리 실험 데이터를 바탕으로 잔존 용량을 참고용으로 추정합니다.</p>
</div>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "nasa_battery_discharge_for_streamlit.csv"

@st.cache_data
def load_data(path_str):
    df = pd.read_csv(path_str)
    required = {"battery_id", "cycle", "ambient_temperature_C", "capacity_Ah"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")
    df = df.dropna(subset=["cycle", "ambient_temperature_C", "capacity_Ah"]).copy()
    df = df[df["capacity_Ah"] > 0]
    # 배터리마다 초기 용량이 다르므로 첫 5회 방전의 중앙값을 기준 용량으로 사용
    baselines = (
        df.sort_values("cycle")
          .groupby("battery_id")["capacity_Ah"]
          .apply(lambda s: s.head(5).median())
    )
    df["baseline_Ah"] = df["battery_id"].map(baselines)
    df = df.dropna(subset=["baseline_Ah"])
    df = df[df["baseline_Ah"] > 0]
    df["soh_pct"] = (df["capacity_Ah"] / df["baseline_Ah"] * 100).clip(0, 130)
    return df

@st.cache_resource
def train_model(path_str):
    df = load_data(path_str)
    X = df[["cycle", "ambient_temperature_C"]]
    y = df["soh_pct"]
    model = RandomForestRegressor(
        n_estimators=220, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    return model, float(df["cycle"].max()), df

try:
    if not DATA_PATH.exists():
        st.error("CSV 파일을 찾을 수 없습니다. main.py와 같은 폴더에 nasa_battery_discharge_for_streamlit.csv를 넣어 주세요.")
        st.stop()
    model, max_cycle, data = train_model(str(DATA_PATH))
except Exception as e:
    st.error(f"데이터를 읽거나 모델을 학습하지 못했습니다: {e}")
    st.stop()

st.markdown('<div class="section-title">1. 보조배터리 사용 정보 입력</div>', unsafe_allow_html=True)
st.caption("제품 라벨을 몰라도 답할 수 있는 항목만 준비했어요. 정확히 모르면 가장 가까운 값으로 선택하세요.")

with st.form("battery_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age_months = st.number_input("사용한 기간 (개월)", min_value=0, max_value=120, value=12, step=1,
                                     help="구매 후 실제 사용한 개월 수")
    with c2:
        cycles_per_week = st.slider("일주일에 완전히 충전·방전한 횟수", 0.0, 14.0, 3.0, 0.5,
                                    help="배터리를 거의 다 사용한 뒤 다시 완전히 충전한 횟수를 대략 입력하세요.")
    with c3:
        ambient_temp = st.slider("주로 사용하는 주변 온도 (℃)", 5, 40, 25,
                                 help="여름철 실내·가방 속 등 평소 사용 환경을 기준으로 선택하세요.")
    st.markdown("---")
    st.markdown("**안전 상태 확인**")
    s1, s2, s3 = st.columns(3)
    with s1:
        swollen = st.checkbox("배터리가 부풀어 올랐거나 외형이 변형됨")
        damaged = st.checkbox("외부 케이스가 깨지거나 심하게 찌그러짐")
    with s2:
        unusual_heat = st.checkbox("사용하지 않을 때도 손대기 어려울 만큼 뜨거움")
        repeated_heat = st.checkbox("충전·사용 중 평소보다 자주 과열됨")
    with s3:
        smell_or_sound = st.checkbox("이상한 냄새, 연기, 소리 등이 발생함")
        charging_issue = st.checkbox("충전이 끊기거나 전원이 비정상적으로 꺼짐")
    submitted = st.form_submit_button("내 보조배터리 상태 확인하기", type="primary", use_container_width=True)

if submitted:
    estimated_cycles = age_months / 12 * 52.18 * cycles_per_week
    estimated_cycles = max(1.0, estimated_cycles) if age_months > 0 and cycles_per_week > 0 else 1.0
    # NASA 데이터 범위 밖 입력은 외삽하지 않고 최대 관측 사이클로 제한
    cycle_for_model = min(estimated_cycles, max_cycle)
    input_df = pd.DataFrame({
        "cycle": [cycle_for_model],
        "ambient_temperature_C": [float(ambient_temp)]
    })
    predicted_soh = float(np.clip(model.predict(input_df)[0], 0, 120))

    # 온도 조건을 고정한 사이클별 예측 곡선을 단조 감소 형태로 보정
    cycle_grid = np.arange(1, int(max_cycle) + 1)
    curve_X = pd.DataFrame({
        "cycle": cycle_grid,
        "ambient_temperature_C": np.full(len(cycle_grid), float(ambient_temp))
    })
    raw_curve = model.predict(curve_X)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    smooth_curve = iso.fit_transform(cycle_grid, raw_curve)
    current_idx = int(np.clip(round(cycle_for_model), 1, len(cycle_grid))) - 1
    predicted_soh = float(smooth_curve[current_idx])

    danger_signs = swollen or damaged or unusual_heat or smell_or_sound
    caution_signs = repeated_heat or charging_issue
    if danger_signs:
        status = "즉시 사용 중단"
        status_color = "#b42318"
        status_bg = "#fff0ef"
        status_message = "외형 변형·심한 발열·이상 냄새 등 위험 징후가 선택되었습니다. 용량 예측과 관계없이 사용 및 충전을 중단하세요."
    elif caution_signs or predicted_soh < 80:
        status = "주의 필요"
        status_color = "#a15c00"
        status_bg = "#fff7e8"
        status_message = "이상 징후 또는 노후화 가능성이 있습니다. 사용을 줄이고 상태를 점검하거나 교체를 고려하세요."
    else:
        status = "입력 기준 특이 징후 없음"
        status_color = "#087443"
        status_bg = "#eaf8f0"
        status_message = "입력한 정보만으로 즉각적인 위험 징후는 확인되지 않았습니다. 안전을 보장하는 판정은 아닙니다."

    # 80% SOH를 참고 기준으로 삼되, NASA 관측 범위 안에서만 잔여 사이클을 계산
    crossing = np.where(smooth_curve <= 80)[0]
    remaining_cycles = None
    remaining_months = None
    if len(crossing):
        threshold_cycle = int(cycle_grid[crossing[0]])
        if threshold_cycle > cycle_for_model:
            remaining_cycles = threshold_cycle - cycle_for_model
            if cycles_per_week > 0:
                remaining_months = remaining_cycles / cycles_per_week / 4.345

    st.markdown("---")
    st.markdown('<div class="section-title">2. 진단 결과</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{status_bg};border:1px solid {status_color}33;border-left:7px solid {status_color};'
        f'padding:18px 20px;border-radius:16px;margin-bottom:16px;">'
        f'<div style="font-size:1.35rem;font-weight:800;color:{status_color};">⚠ {status}</div>'
        f'<div style="color:#334155;margin-top:6px;">{status_message}</div></div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("예상 잔존 용량 수준 (SOH)", f"{predicted_soh:.1f}%")
    m2.metric("추정 사용 누적 사이클", f"{estimated_cycles:,.0f}회")
    if remaining_months is not None:
        m3.metric("80% 도달까지 참고 기간", f"약 {remaining_months:.1f}개월")
    else:
        m3.metric("80% 도달까지 참고 기간", "산출 불가")
    st.caption("SOH(건강 상태)는 해당 배터리의 초기 기준 용량 대비 현재 용량의 비율입니다. 80%는 참고용 성능 기준이며, 안전 사용 기한이나 화재 발생 시점을 뜻하지 않습니다.")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("**사이클에 따른 잔존 용량 추정 곡선**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cycle_grid, y=smooth_curve, mode="lines",
                                 name="모델 추정 SOH", line=dict(color="#167d78", width=3),
                                 fill="tozeroy", fillcolor="rgba(22,125,120,.08)"))
        fig.add_hline(y=80, line_dash="dash", line_color="#d97706",
                      annotation_text="참고 기준 80%")
        fig.add_trace(go.Scatter(x=[cycle_for_model], y=[predicted_soh], mode="markers",
                                 name="현재 입력값", marker=dict(size=13, color="#e4572e",
                                 line=dict(width=2, color="white"))))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=25,b=10),
                          xaxis_title="누적 방전 사이클(회)", yaxis_title="잔존 용량 수준(SOH, %)",
                          yaxis=dict(range=[0, 120]), plot_bgcolor="white",
                          paper_bgcolor="white", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**현재 상태 한눈에 보기**")
        gauge_color = "#b42318" if danger_signs else ("#d97706" if predicted_soh < 80 or caution_signs else "#16845b")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=predicted_soh, number={"suffix":"%","font":{"size":34}},
            gauge={"axis":{"range":[0,120]}, "bar":{"color":gauge_color},
                   "steps":[{"range":[0,60],"color":"#fee2e2"},
                            {"range":[60,80],"color":"#fef3c7"},
                            {"range":[80,120],"color":"#dcfce7"}],
                   "threshold":{"line":{"color":"#334155","width":3},"thickness":.8,"value":80}}
        ))
        gauge.update_layout(height=300, margin=dict(l=20,r=20,t=35,b=10),
                            paper_bgcolor="white")
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(
            f'<div class="panel"><div class="metric-label">입력된 주변 온도</div>'
            f'<div class="metric-value">{ambient_temp}℃</div>'
            f'<div class="small-note">평소 사용 환경을 기준으로 한 모델 입력값</div></div>',
            unsafe_allow_html=True
        )
