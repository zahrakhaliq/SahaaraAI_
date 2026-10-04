import streamlit as st

from agents import build_graph

st.set_page_config(page_title="Sahaara AI", page_icon="🩺", layout="centered")


@st.cache_resource
def graph():
    return build_graph()


st.title("🩺 Sahaara AI")
st.caption("Early health support in English, Urdu and Roman Urdu. Apni takleef apne alfaaz mein likhein.")
st.info("Sahaara AI gives early health information. It does not diagnose and does not replace a doctor. "
        "Emergency: call 1122 / 115.", icon="ℹ️")

LANGS = {"Auto-detect": None, "English": "en", "اردو": "ur", "Roman Urdu": "roman_ur"}
AGES = {"Adult": "adult", "Older adult (65+)": "older_adult", "Child": "child",
        "Infant (under 1 year)": "infant", "Baby (under 3 months)": "newborn"}

with st.form("case"):
    text = st.text_area("What are you feeling? / Aap ko kya takleef hai?", height=120,
                        placeholder="Mujhe subah se chakkar aa rahe hain aur bohat kamzori hai...")
    c1, c2 = st.columns(2)
    age = c1.selectbox("Patient", list(AGES))
    lang = c2.selectbox("Reply language", list(LANGS))
    with st.expander("Optional: measured values (leave 0 if not measured)"):
        m1, m2, m3 = st.columns(3)
        bp_sys = m1.number_input("BP top (systolic)", 0, 300, 0)
        bp_dia = m2.number_input("BP bottom (diastolic)", 0, 200, 0)
        glucose = m3.number_input("Glucose (mg/dL)", 0, 900, 0)
        m4, m5 = st.columns(2)
        temp = m4.number_input("Temperature (°C)", 0.0, 45.0, 0.0, step=0.1)
        spo2 = m5.number_input("Oxygen SpO2 (%)", 0, 100, 0)
    go = st.form_submit_button("Get guidance", type="primary", use_container_width=True)

if go:
    if not text.strip():
        st.warning("Please describe what you are feeling.")
        st.stop()
    measures = {k: float(v) for k, v in {"bp_sys": bp_sys, "bp_dia": bp_dia, "glucose": glucose,
                                         "temp_c": temp, "spo2": spo2}.items() if v}
    with st.spinner("Sahaara agents are working..."):
        out = graph().invoke({"text": text, "age_group": AGES[age], "lang_pref": LANGS[lang],
                              "measures": measures, "trace": []})
    (st.error if out["kind"] == "emergency" else st.markdown)(out["final"])
    if out["kind"] == "guidance":
        st.download_button("📋 Download case report for your doctor", out["report"],
                           file_name="sahaara_case_report.txt")
        with st.expander("Case report preview"):
            st.code(out["report"], language=None)
    with st.expander("🤖 What the agents did"):
        for step in out["trace"]:
            st.write("•", step)
