            height=370,
            xaxis_title="누적 방전 사이클 (회)",
            yaxis_title="잔존 용량 수준 SOH (%)",
            yaxis_range=[0, 120],
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(line_chart, use_container_width=True)

    with gauge_col:
        st.markdown("#### 🔋 잔존 용량 게이지")
        gauge_color = (
            "#b42318" if critical_sign
            else "#d97706" if caution_sign or predicted_soh < 80
            else "#16845b"
        )
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_soh,
                number={"suffix": "%", "font": {"size": 34}},
                gauge={
                    "axis": {"range": [0, 120]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 60], "color": "#fee2e2"},
                        {"range": [60, 80], "color": "#fef3c7"},
                        {"range": [80, 120], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "#334155", "width": 3},
                        "thickness": 0.8,
                        "value": 80,
                    },
                },
            )
        )
        gauge.update_layout(
            height=330,
            margin=dict(l=20, r=20, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gauge, use_container_width=True)

        st.metric("입력한 평소 주변 온도", f"{usual_temperature} ℃")

    if estimated_cycles > max_cycle:
        st.warning(
            f"입력으로 계산한 누적 사이클({estimated_cycles:,.0f}회)이 "
            f"NASA 데이터의 최대 관측 사이클({max_cycle:,}회)을 초과합니다. "
            "예측 입력은 데이터 범위의 끝값으로 제한되므로 결과의 신뢰도가 낮을 수 있습니다."
        )

    st.markdown("#### 🧯 안전 행동 안내")
    if critical_sign:
        st.error(
            "사용과 충전을 즉시 중단하세요. 안전하게 할 수 있는 경우 전원과 케이블을 분리하고 "
            "가연물에서 멀리하세요. 부풀거나 손상된 배터리를 누르거나 뚫거나 분해하지 마세요. "
            "연기나 불꽃이 발생하면 접근하지 말고 대피한 뒤 119에 신고하세요."
        )
    elif caution_sign:
        st.warning(
            "충전 중 자리를 비우지 말고 침구나 종이 등 가연물 위에서 충전하지 마세요. "
            "문제가 반복되면 사용을 중단하고 제조사 또는 폐배터리 수거 기관에 문의하세요."
        )
    else:
        st.info(
            "인증된 충전기와 케이블을 사용하고 직사광선, 고온의 차량 내부, 침구 위 충전을 피하세요. "
            "외형 변화나 비정상적인 발열이 생기면 즉시 사용을 중단하세요."
        )

    st.markdown("#### 📊 입력한 조건과 비슷한 NASA 실험 데이터")
    st.caption("아래 산점도는 원본 실험 데이터이며, 사용자의 보조배터리를 직접 측정한 값은 아닙니다.")
    sample = battery_data.sample(min(1200, len(battery_data)), random_state=42)
    scatter = px.scatter(
        sample,
        x="cycle",
        y="soh_percent",
        color="ambient_temperature_C",
        hover_data=["battery_id", "capacity_Ah"],
        labels={
            "cycle": "방전 사이클 (회)",
            "soh_percent": "잔존 용량 수준 SOH (%)",
            "ambient_temperature_C": "주변 온도 (℃)",
            "battery_id": "배터리 ID",
            "capacity_Ah": "방전 용량 (Ah)",
        },
        color_continuous_scale="Teal",
    )
    scatter.update_layout(
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=25, b=10),
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.info(
        "이 앱은 NASA PCoE의 실험용 리튬이온 셀 데이터를 활용한 교육·참고용 도구입니다. "
        "일반 보조배터리의 셀 구성, 보호회로, 제조 품질 및 실제 사용 이력과 차이가 있으므로 "
        "정확한 수명·화재 확률·안전성을 판정하지 않습니다. 위험 경고는 사용자가 체크한 "
        "외형 및 이상 징후를 바탕으로 한 예방 안내이며 화재 예측 모델의 결과가 아닙니다."
    )

else:
    st.markdown("### 앱 이용 순서")
    info1, info2, info3 = st.columns(3)
    with info1:
        st.info("**1. 사용 기간 입력**\n\n구매 후 몇 개월 사용했는지 입력합니다.")
    with info2:
        st.info("**2. 사용 환경 입력**\n\n주간 충·방전 횟수와 주변 온도를 선택합니다.")
    with info3:
        st.info("**3. 안전 상태 확인**\n\n부풀음·발열·냄새 등 이상 징후를 체크합니다.")

    st.markdown("### 이 앱이 보여주는 결과")
    st.write(
        "- 잔존 용량 수준을 게이지로 표시\n"
        "- 누적 충·방전 횟수에 따른 용량 변화 그래프\n"
        "- 80% 도달까지의 참고 기간\n"
        "- 이상 징후가 있을 때 눈에 띄는 안전 경고"
    )

st.divider()
st.caption(
    "BATTERY CARE | NASA PCoE 데이터 기반 교육용 프로젝트 · "
    "앱의 예측값은 실제 제품의 안전성 인증이나 사용 가능 기간을 보장하지 않습니다."
