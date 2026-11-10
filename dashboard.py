from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NASA RL Auto-Scaling Dashboard", layout="wide")
st.title("NASA-Driven Reinforcement Learning Auto-Scaling")
st.caption("Three-tier cloud simulator | NASA HTTP workload trace | Q-Learning, PPO, TRPO and HPA baseline")

summary_path = Path("results/summary.csv")
if not summary_path.exists():
    st.warning("No experiment results found. Run `python prepare_data.py` and then `python train.py` first.")
    st.stop()

summary = pd.read_csv(summary_path)
c1, c2, c3 = st.columns(3)
best_latency = summary.loc[summary.mean_response_time_ms.idxmin()]
best_cost = summary.loc[summary.mean_resource_cost.idxmin()]
best_sla = summary.loc[summary.sla_violation_rate_pct.idxmin()]
c1.metric("Lowest mean response time", best_latency.algorithm, f"{best_latency.mean_response_time_ms:.1f} ms")
c2.metric("Lowest estimated cost", best_cost.algorithm, f"{best_cost.mean_resource_cost:.3f}")
c3.metric("Lowest SLA violation", best_sla.algorithm, f"{best_sla.sla_violation_rate_pct:.2f}%")

st.subheader("Algorithm comparison")
st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
st.bar_chart(summary.set_index("algorithm")[["mean_response_time_ms"]])
st.bar_chart(summary.set_index("algorithm")[["mean_resource_cost"]])

st.subheader("SLA Violation Rate Comparison")

sla_data = summary[["algorithm", "sla_violation_rate_pct"]].copy()

st.dataframe(
    sla_data,
    use_container_width=True,
    hide_index=True
)

if (sla_data["sla_violation_rate_pct"] == 0).all():
    st.success("All four algorithms maintained 0% SLA violations during evaluation.")
else:
    st.bar_chart(
        sla_data.set_index("algorithm")["sla_violation_rate_pct"]
    )


st.subheader("Generated experiment graphs")
for filename, caption in [
    ("response_time_comparison.png", "Response time over evaluation intervals"),
    ("resource_cost_comparison.png", "Estimated resource cost"),

]:
    path = Path("results") / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)

st.info("Interpretation note: NASA provides HTTP request traffic. CPU utilization, response time, SLA violations, replica counts and resource cost are simulator-derived estimates, not NASA-measured cloud telemetry.")
