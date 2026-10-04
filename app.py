import html

import streamlit as st

from agents import HEAD, build_graph

st.set_page_config(page_title="Sahaara AI", page_icon="🩺", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none !important;}
.block-container {max-width: 820px; padding-top: 1.2rem; padding-bottom: 3rem;}
.topbar {display: flex; justify-content: space-between; align-items: center; padding: 10px 0 14px;
         border-bottom: 1px solid rgba(128,128,128,.25); margin-bottom: 22px;}
.brand {font-size: 1.35rem; font-weight: 700; letter-spacing: -.01em;}
.brand span {color: #0E7490;}
.meta {font-size: .8rem; opacity: .65;}
.title {font-size: 1.9rem; font-weight: 700; line-height: 1.25; margin: 6px 0 4px; letter-spacing: -.02em;}
.lead {opacity: .72; margin-bottom: 18px; font-size: 1rem;}
.notice {border-left: 3px solid #DC2626; background: rgba(220,38,38,.08); padding: 9px 14px; border-radius: 6px;
         font-size: .87rem; margin-bottom: 22px;}
.label {font-size: .74rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .6; margin: 14px 0 6px;}
.verdict {border: 1px solid; border-left-width: 6px; border-radius: 10px; padding: 16px 20px; margin: 22px 0 14px;}
.verdict .t {font-size: 1.25rem; font-weight: 700;} .verdict .s {opacity: .85; margin-top: 2px; font-size: .95rem;}
.v-green {border-color: #059669; background: rgba(5,150,105,.10);}
.v-yellow {border-color: #D97706; background: rgba(217,119,6,.10);}
.v-red {border-color: #DC2626; background: rgba(220,38,38,.10);}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: 14px 18px; background: rgba(128,128,128,.05);
       margin-bottom: 12px; height: 100%;}
.card h4 {margin: 0 0 8px; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .6;}
.card p, .card li {unicode-bidi: plaintext; text-align: start; line-height: 1.6; margin: 0 0 4px; font-size: .96rem;}
.card ul {margin: 0; padding-left: 1.1rem;}
.card.warn {border-color: rgba(220,38,38,.45); background: rgba(220,38,38,.06);}
.card.next {border-color: rgba(14,116,144,.55); background: rgba(14,116,144,.10);}
.callbtn {display: inline-block; background: #DC2626; color: #fff !important; padding: 10px 24px; border-radius: 8px;
          font-weight: 700; text-decoration: none; margin: 4px 0 10px;}
.chips span {display: inline-block; border: 1px solid rgba(128,128,128,.3); padding: 2px 11px; border-radius: 999px;
             font-size: .78rem; margin: 0 6px 8px 0; opacity: .85;}
.step {display: flex; gap: 12px; padding: 7px 0; border-bottom: 1px solid rgba(128,128,128,.2); font-size: .9rem;}
.step .n {font-weight: 700; color: #0E7490; min-width: 20px;}
.fine {font-size: .78rem; opacity: .6; margin-top: 18px;}
div[data-testid="stFormSubmitButton"] button {background: #0E7490; color: #fff; border: 0; border-radius: 8px;
    font-weight: 600; padding: .6rem 1rem;}
div[data-testid="stFormSubmitButton"] button:hover {background: #155E75; color: #fff;}
div[data-testid="stForm"] {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 18px 20px;}
div.stButton > button {border-radius: 8px; font-size: .85rem;}
textarea {border-radius: 8px !important;}
</style>
<div class="topbar"><div class="brand">🩺 Sahaara<span> AI</span></div>
<div class="meta">Early health support &nbsp;·&nbsp; English &nbsp;·&nbsp; اردو &nbsp;·&nbsp; Roman Urdu</div></div>
<div class="title">Describe how you are feeling</div>
<div class="lead">Tell us in your own words. Sahaara checks for warning signs, suggests safe next steps and prepares a report for your doctor.</div>
<div class="notice"><b>Emergency?</b> Call <b>1122 / 115</b> or go to the nearest hospital. Sahaara AI does not diagnose and does not replace a doctor.</div>
""", unsafe_allow_html=True)


@st.cache_resource
def graph():
    return build_graph()


EXAMPLES = {
    "Dizziness & weakness": "Mujhe subah se chakkar aa rahe hain aur bohat kamzori hai",
    "Low sugar": "Mera sugar low lag raha hai, paseena aa raha hai aur haath kaanp rahe hain",
    "Fever": "I have had fever and headache for 2 days",
    "Blood pressure": "Mera BP high lag raha hai, sar bohat dard kar raha hai",
    "Fainting": "Meri behen achanak behosh ho gayi thi",
    "Chest pain": "Mujhe seene mein dard hai aur saans lene mein mushkil ho rahi hai",
}


def use_example(t):
    st.session_state["text"] = t


st.markdown('<div class="label">Quick examples</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, (label, phrase) in enumerate(EXAMPLES.items()):
    cols[i % 3].button(label, key=f"ex{i}", on_click=use_example, args=(phrase,), use_container_width=True)

LANGS = {"Auto-detect": None, "English": "en", "اردو": "ur", "Roman Urdu": "roman_ur"}
AGES = {"Adult": "adult", "Older adult (65+)": "older_adult", "Child": "child",
        "Infant (under 1 year)": "infant", "Baby (under 3 months)": "newborn"}

with st.form("case"):
    text = st.text_area("Your symptoms", key="text", height=110,
                        placeholder="e.g. Mujhe subah se chakkar aa rahe hain aur bohat kamzori hai")
    c1, c2 = st.columns(2)
    age = c1.selectbox("Patient", list(AGES))
    lang = c2.selectbox("Reply language", list(LANGS))
    with st.expander("Measured values (optional, leave 0 if not measured)"):
        m1, m2, m3 = st.columns(3)
        bp_sys = m1.number_input("BP systolic (top)", 0, 300, 0)
        bp_dia = m2.number_input("BP diastolic (bottom)", 0, 200, 0)
        glucose = m3.number_input("Glucose (mg/dL)", 0, 900, 0)
        m4, m5 = st.columns(2)
        temp = m4.number_input("Temperature (°C)", 0.0, 45.0, 0.0, step=0.1)
        spo2 = m5.number_input("Oxygen SpO2 (%)", 0, 100, 0)
    go = st.form_submit_button("Get guidance", use_container_width=True)

TITLE = {"green": "Lower concern", "yellow": "Needs medical evaluation", "red": "Urgent: seek help now"}
SUB = {"green": "This can likely be managed with care at home while you monitor.",
       "yellow": "Please arrange to be seen by a healthcare professional.",
       "red": "Get medical help immediately."}


def card(title, body, cls=""):
    e = html.escape
    if isinstance(body, list):
        body = "<ul>" + "".join(f"<li>{e(str(x))}</li>" for x in body) + "</ul>"
    else:
        body = f"<p>{e(str(body))}</p>"
    return f'<div class="card {cls}"><h4>{e(title)}</h4>{body}</div>'


if go:
    if not text.strip():
        st.warning("Please describe what you are feeling.")
        st.stop()
    measures = {k: float(v) for k, v in {"bp_sys": bp_sys, "bp_dia": bp_dia, "glucose": glucose,
                                         "temp_c": temp, "spo2": spo2}.items() if v}
    with st.spinner("Analysing your symptoms..."):
        out = graph().invoke({"text": text, "age_group": AGES[age], "lang_pref": LANGS[lang],
                              "measures": measures, "trace": []})
    risk = out["risk"] if out["kind"] != "fallback" else "yellow"
    st.markdown(f'<div class="verdict v-{risk}"><div class="t">{TITLE[risk]}</div><div class="s">{SUB[risk]}</div></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="chips"><span>Topic: {html.escape(str(out.get("topic", "-")))}</span>'
                f'<span>Language: {html.escape(str(out.get("language", "-")))}</span>'
                f'<span>{len(out["trace"])} agent steps</span></div>', unsafe_allow_html=True)

    if out["kind"] == "emergency":
        st.markdown('<a class="callbtn" href="tel:1122">Call 1122</a>', unsafe_allow_html=True)
        st.markdown(out["final"])
    elif out["kind"] == "fallback":
        st.warning(out["final"])
    else:
        a, h = out["advice"], HEAD[out["language"]]
        st.markdown(card(h[0], a.get("understood", "")) + card(h[1], a.get("relevant", "")), unsafe_allow_html=True)
        if a.get("to_check"):
            st.markdown(card(h[2], a["to_check"]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown(card(h[3], a.get("do_now", [])), unsafe_allow_html=True)
        c2.markdown(card(h[4].split(":")[0], a.get("warning_signs", []), "warn"), unsafe_allow_html=True)
        st.markdown(card(h[5], a.get("next_step", ""), "next"), unsafe_allow_html=True)
        st.download_button("Download case report for your doctor", out["report"],
                           file_name="sahaara_case_report.txt", use_container_width=True)
        with st.expander("Preview case report"):
            st.code(out["report"], language=None)

    with st.expander("Agent activity"):
        for i, step in enumerate(out["trace"], 1):
            name, _, note = step.partition(":")
            st.markdown(f'<div class="step"><div class="n">{i}</div><div><b>{html.escape(name)}</b>'
                        f' &nbsp;{html.escape(note.strip())}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="fine">Sahaara AI provides general early health information only. '
                'It cannot diagnose conditions or replace professional medical care.</div>', unsafe_allow_html=True)
