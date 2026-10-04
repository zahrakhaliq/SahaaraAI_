import streamlit as st

from agents import BANNER, build_graph

st.set_page_config(page_title="Sahaara AI", page_icon="🩺", layout="centered")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 780px;}
.hero {background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%); color: #fff; padding: 28px 26px;
       border-radius: 20px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(15,118,110,.25);}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff;}
.hero p {margin: 6px 0 0; opacity: .95; font-size: 1.02rem;}
.hero .tag {display: inline-block; background: rgba(255,255,255,.2); padding: 3px 12px; border-radius: 999px;
            font-size: .8rem; margin-top: 12px; margin-right: 6px;}
.sos {background: #FEF2F2; border: 1px solid #FCA5A5; color: #991B1B; padding: 10px 14px; border-radius: 12px;
      font-size: .92rem; margin-bottom: 14px;}
.banner {padding: 16px 20px; border-radius: 16px; border-left: 8px solid; margin: 10px 0 14px;}
.banner .t {font-size: 1.35rem; font-weight: 700;} .banner .s {font-size: .95rem; margin-top: 2px;}
.b-green {background: #ECFDF5; border-color: #10B981; color: #065F46;}
.b-yellow {background: #FFFBEB; border-color: #F59E0B; color: #92400E;}
.b-red {background: #FEF2F2; border-color: #EF4444; color: #991B1B;}
.callbtn {display: inline-block; background: #DC2626; color: #fff !important; padding: 10px 22px; border-radius: 999px;
          font-weight: 700; font-size: 1.1rem; text-decoration: none; margin: 4px 0 12px;}
.chips span {display: inline-block; background: rgba(100,116,139,.15); padding: 3px 12px; border-radius: 999px;
             font-size: .82rem; margin: 0 6px 6px 0;}
.step {display: flex; gap: 12px; align-items: flex-start; padding: 8px 0; border-bottom: 1px dashed rgba(100,116,139,.3);}
.step .n {background: #0F766E; color: #fff; min-width: 26px; height: 26px; border-radius: 50%; text-align: center;
          line-height: 26px; font-size: .8rem; font-weight: 700;}
.res p, .res li {unicode-bidi: plaintext; text-align: start; line-height: 1.65;}
div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] button {
    background: #0F766E; color: #fff; border: 0; border-radius: 12px; font-weight: 700; padding: .65rem 1rem;}
div[data-testid="stFormSubmitButton"] button:hover {background: #115E59; color: #fff;}
textarea {border-radius: 12px !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🩺 Sahaara AI</h1>
  <p>Apni takleef apne alfaaz mein batayein. Describe how you feel, and get safe next-step guidance.</p>
  <span class="tag">English</span><span class="tag">اردو</span><span class="tag">Roman Urdu</span>
  <span class="tag">Multi-agent AI</span>
</div>
<div class="sos">🚨 <b>Emergency?</b> Call <b>1122 / 115</b> or go to the nearest hospital. Sahaara AI does not diagnose and does not replace a doctor.</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("How it works")
    st.markdown("1. 📝 **Intake** reads your words\n2. 🚦 **Triage** checks for danger\n3. 💡 **Advice** gives safe steps\n"
                "4. 📄 **Referral** prepares a doctor report\n5. ✅ **Safety Checker** reviews the answer")
    st.divider()
    st.caption("Pak Angels Generative & Agentic AI Training, Cohort 11")


@st.cache_resource
def graph():
    return build_graph()


EXAMPLES = {
    "😵 Chakkar & weakness": "Mujhe subah se chakkar aa rahe hain aur bohat kamzori hai",
    "🍬 Sugar low": "Mera sugar low lag raha hai, paseena aa raha hai aur haath kaanp rahe hain",
    "🌡️ Fever": "I have had fever and headache for 2 days",
    "❤️ BP": "Mera BP high lag raha hai, sar bohat dard kar raha hai",
    "😶‍🌫️ Fainting": "Meri behen achanak behosh ho gayi thi",
    "🆘 Chest pain": "Mujhe seene mein dard hai aur saans lene mein mushkil ho rahi hai",
}


def use_example(t):
    st.session_state["text"] = t


st.markdown("##### Try an example")
cols = st.columns(3)
for i, (label, phrase) in enumerate(EXAMPLES.items()):
    cols[i % 3].button(label, key=f"ex{i}", on_click=use_example, args=(phrase,), use_container_width=True)

LANGS = {"Auto-detect": None, "English": "en", "اردو": "ur", "Roman Urdu": "roman_ur"}
AGES = {"Adult": "adult", "Older adult (65+)": "older_adult", "Child": "child",
        "Infant (under 1 year)": "infant", "Baby (under 3 months)": "newborn"}

with st.form("case"):
    text = st.text_area("What are you feeling? / Aap ko kya takleef hai?", key="text", height=120,
                        placeholder="Mujhe subah se chakkar aa rahe hain aur bohat kamzori hai...")
    c1, c2 = st.columns(2)
    age = c1.selectbox("👤 Patient", list(AGES))
    lang = c2.selectbox("🌐 Reply language", list(LANGS))
    with st.expander("🩸 Optional: measured values (leave 0 if not measured)"):
        m1, m2, m3 = st.columns(3)
        bp_sys = m1.number_input("BP top (systolic)", 0, 300, 0)
        bp_dia = m2.number_input("BP bottom (diastolic)", 0, 200, 0)
        glucose = m3.number_input("Glucose (mg/dL)", 0, 900, 0)
        m4, m5 = st.columns(2)
        temp = m4.number_input("Temperature (°C)", 0.0, 45.0, 0.0, step=0.1)
        spo2 = m5.number_input("Oxygen SpO2 (%)", 0, 100, 0)
    go = st.form_submit_button("Get guidance  ➜", use_container_width=True)

SUB = {"green": "You can likely manage this with care at home while monitoring.",
       "yellow": "Please arrange to see a healthcare professional.",
       "red": "Get medical help immediately."}

if go:
    if not text.strip():
        st.warning("Please describe what you are feeling.")
        st.stop()
    measures = {k: float(v) for k, v in {"bp_sys": bp_sys, "bp_dia": bp_dia, "glucose": glucose,
                                         "temp_c": temp, "spo2": spo2}.items() if v}
    with st.spinner("Sahaara agents are working..."):
        out = graph().invoke({"text": text, "age_group": AGES[age], "lang_pref": LANGS[lang],
                              "measures": measures, "trace": []})
    risk = out["risk"]
    st.markdown(f'<div class="banner b-{risk}"><div class="t">{BANNER[risk]}</div><div class="s">{SUB[risk]}</div></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="chips"><span>🏷️ Topic: {out.get("topic", "-")}</span>'
                f'<span>🌐 Language: {out.get("language", "-")}</span>'
                f'<span>🤖 {len(out["trace"])} agent steps</span></div>', unsafe_allow_html=True)

    with st.container(border=True):
        if out["kind"] == "emergency":
            st.markdown('<a class="callbtn" href="tel:1122">📞 Call 1122</a>', unsafe_allow_html=True)
            st.markdown(out["final"])
        elif out["kind"] == "guidance":
            body = out["final"].split("\n\n", 1)[1]  # banner is shown above
            st.markdown(f'<div class="res">\n\n{body}\n\n</div>', unsafe_allow_html=True)
        else:
            st.warning(out["final"])

    if out["kind"] == "guidance":
        st.download_button("📋 Download case report for your doctor", out["report"],
                           file_name="sahaara_case_report.txt", use_container_width=True)
        with st.expander("Preview case report"):
            st.code(out["report"], language=None)
    with st.expander("🤖 What the agents did"):
        for i, step in enumerate(out["trace"], 1):
            name, _, note = step.partition(":")
            st.markdown(f'<div class="step"><div class="n">{i}</div><div><b>{name}</b><br>{note.strip()}</div></div>',
                        unsafe_allow_html=True)
