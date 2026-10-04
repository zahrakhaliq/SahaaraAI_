import html

import streamlit as st

import agents
from agents import HEAD, build_graph

st.set_page_config(page_title="Sahaara AI", page_icon="🩺", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none !important;}
header[data-testid="stHeader"] {display: none !important;}
.block-container {max-width: 820px; padding-top: 2rem; padding-bottom: 3rem;}
.topbar {display: flex; justify-content: space-between; align-items: center; padding: 10px 0 14px;
         border-bottom: 1px solid rgba(128,128,128,.25); margin-bottom: 22px;}
.brand {font-size: 1.35rem; font-weight: 700; letter-spacing: -.01em;}
.brand span {color: #0E7490;}
.meta {font-size: .8rem; opacity: .65;}
.title {font-size: 1.9rem; font-weight: 700; line-height: 1.25; margin: 6px 0 4px; letter-spacing: -.02em;}
.lead {opacity: .72; margin-bottom: 18px; font-size: 1rem;}
.notice {border-left: 3px solid #DC2626; background: rgba(220,38,38,.08); padding: 9px 14px; border-radius: 6px;
         font-size: .87rem; margin-bottom: 12px;}
.proto {border-left: 3px solid #D97706; background: rgba(217,119,6,.12); padding: 9px 14px; border-radius: 6px;
        font-size: .87rem; margin: 8px 0 14px;}
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
.logo-wrap {text-align: center; padding: 8px 0 22px; border-bottom: 1px solid rgba(128,128,128,.25); margin-bottom: 26px;}
.logo-wrap .name {font-size: 2.4rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; margin-top: 10px;}
.logo-wrap .name span {color: #0E7490;}
.logo-wrap .urdu {font-size: 1.5rem; color: #0E7490; margin-top: 2px;}
.logo-wrap .tagline {font-size: 1.02rem; opacity: .75; margin-top: 4px;}
.logo-wrap .langs {font-size: .8rem; opacity: .55; margin-top: 8px;}
.title {text-align: center;}
.lead {text-align: center;}
</style>
""", unsafe_allow_html=True)

LOGO = """<div class="logo-wrap"><svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><rect width="64" height="64" rx="16" fill="#0E7490"/><path d="M26 12h12v12h12v12H38v12H26V36H14V24h12z" fill="#fff"/><path d="M12 54c10 6 30 6 40 0" stroke="#A5F3FC" stroke-width="3" fill="none" stroke-linecap="round"/></svg><div class="name">Sahaara <span>AI</span></div><div class="urdu">سہارا</div><div class="tagline">Your first step to safe care</div><div class="langs">English &nbsp;·&nbsp; اردو &nbsp;·&nbsp; Roman Urdu</div></div>"""
st.markdown(LOGO, unsafe_allow_html=True)
st.caption(f"Build {agents.VERSION} · live Groq model `{agents.resolve_model()}`")
st.markdown(f"""<div class="proto">{html.escape(agents.DISCLAIMER)}</div>
<div class="notice"><b>Emergency?</b> Call <b>1122 / 115</b> or go to the nearest hospital.</div>
""", unsafe_allow_html=True)

view = st.radio("Open", ["Patient guidance", "Health worker queue"], horizontal=True, label_visibility="collapsed")


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


LANGS = {"Auto-detect": None, "English": "en", "اردو": "ur", "Roman Urdu": "roman_ur"}
AGES = {"Adult": "adult", "Older adult (65+)": "older_adult", "Child": "child",
        "Infant (under 1 year)": "infant", "Baby (under 3 months)": "newborn"}

if view == "Health worker queue":
    rows = list(reversed(agents.list_cases()))
    st.markdown('<div class="title">Referral queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="lead">Every finished case is appended to <b>data/referrals.csv</b> on this server. '
                'Update the status as you contact the patient or clinic.</div>', unsafe_allow_html=True)
    if not rows:
        st.info("No cases yet. They show up here after someone gets guidance or an emergency result.")
    else:
        st.dataframe(
            [{k: r.get(k, "") for k in ("created", "id", "status", "risk", "topic", "city", "facility", "referral")} for r in rows],
            use_container_width=True, hide_index=True)
        if agents.QUEUE_PATH.exists():
            st.download_button("Download CSV", agents.QUEUE_PATH.read_bytes(),
                               file_name="sahaara_referrals.csv", use_container_width=True)
        st.markdown('<div class="label">Update status</div>', unsafe_allow_html=True)
        with st.form("queue_status"):
            picks = []
            for r in rows[:12]:
                left, right = st.columns([3, 1])
                left.markdown(f"**{html.escape(r['id'])}** · {html.escape(r.get('risk', ''))} · "
                              f"{html.escape(r.get('topic', ''))} · {html.escape(r.get('city') or 'city not set')}")
                current = r.get("status") if r.get("status") in agents.STATUSES else "new"
                picks.append((r["id"], right.selectbox("Status", agents.STATUSES,
                                                       index=agents.STATUSES.index(current), key=f"st_{r['id']}",
                                                       label_visibility="collapsed")))
            if st.form_submit_button("Save statuses", use_container_width=True):
                for case_id, status in picks:
                    agents.update_status(case_id, status)
                st.rerun()
        with st.expander("Case reports"):
            for r in rows[:12]:
                st.markdown(f"**{html.escape(r['id'])}**")
                st.code(r.get("report") or "", language=None)
    st.markdown(f'<div class="fine">{html.escape(agents.DISCLAIMER)}</div>', unsafe_allow_html=True)
    st.stop()

st.markdown("""<div class="title">Describe how you are feeling</div>
<div class="lead">Tell us in your own words. If duration or city is missing, Sahaara asks before it writes guidance. A facility is looked up for the referral.</div>
""", unsafe_allow_html=True)
st.markdown('<div class="label">Quick examples</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, (label, phrase) in enumerate(EXAMPLES.items()):
    cols[i % 3].button(label, key=f"ex{i}", on_click=use_example, args=(phrase,), use_container_width=True)

with st.form("case"):
    text = st.text_area("Your symptoms", key="text", height=110,
                        placeholder="e.g. Mujhe subah se chakkar aa rahe hain aur bohat kamzori hai")
    c1, c2, c3 = st.columns(3)
    age = c1.selectbox("Patient", list(AGES))
    lang = c2.selectbox("Reply language", list(LANGS))
    city = c3.text_input("City (optional)", placeholder="Lahore")
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


def payload(text, age, lang, city, measures, **extra):
    data = {"text": text, "age_group": AGES[age], "lang_pref": LANGS[lang], "city": (city or "").strip(),
            "measures": measures, "trace": []}
    data.update(extra)
    return data


def show_trace(steps):
    for i, step in enumerate(steps or [], 1):
        name, _, note = step.partition(":")
        st.markdown(f'<div class="step"><div class="n">{i}</div><div><b>{html.escape(name)}</b>'
                    f' &nbsp;{html.escape(note.strip())}</div></div>', unsafe_allow_html=True)


def show_result(out):
    if out.get("source") == "groq":
        st.success(f"Live Groq response · {out.get('model_used') or agents.current_model()}")
    elif out.get("kind") == "emergency" and out.get("model_used") and not out.get("llm_error"):
        st.success(f"Live Groq intake · {out['model_used']}. Emergency wording below is fixed text.")
    elif out.get("llm_error") and out.get("kind") == "emergency":
        st.warning("Groq intake did not answer (" + out["llm_error"] + "). Emergency rules still matched, so urgent instructions are shown.")
    elif out.get("llm_error"):
        retired = "llama-3.3-70b" in out["llm_error"] or "decommission" in out["llm_error"].lower()
        hint = (" The model llama-3.3-70b-versatile was shut down on 16 Aug 2026. This build calls openai/gpt-oss-120b."
                if retired else "")
        st.error("Groq did not complete this request, so draft standard guidance is shown where a model answer was required. "
                 f"Details: `{out['llm_error']}`.{hint} Check GROQ_API_KEY under Settings → Secrets.")
    elif out.get("kind") == "guidance" and out.get("source") != "groq":
        st.warning("Draft standard guidance is shown. It has not been clinically reviewed.")
    if out.get("safety_note"):
        st.info(out["safety_note"])
    risk = out["risk"]
    title, sub = TITLE[risk], SUB[risk]
    st.markdown(f'<div class="verdict v-{risk}"><div class="t">{html.escape(title)}</div>'
                f'<div class="s">{html.escape(sub)}</div></div>', unsafe_allow_html=True)
    fac = out.get("facility") or {}
    via = "Groq tool" if fac.get("via") == "groq" else "local directory"
    st.markdown(f'<div class="chips"><span>Topic: {html.escape(str(out.get("topic", "-")))}</span>'
                f'<span>Language: {html.escape(str(out.get("language", "-")))}</span>'
                f'<span>Source: {html.escape(str(out.get("source") or out.get("kind")))}</span>'
                f'<span>Facility lookup: {html.escape(via)}</span>'
                f'<span>{len(out.get("trace") or [])} agent steps</span></div>', unsafe_allow_html=True)
    if fac.get("chosen"):
        lines = [fac["chosen"] + (f" — {fac['chosen_city'].title()}" if fac.get("chosen_city") else "")]
        if fac.get("level"):
            lines.append(fac["level"])
        if fac.get("note"):
            lines.append(fac["note"])
        others = [m["name"] for m in fac.get("matches") or [] if m.get("name") != fac.get("chosen")]
        if others:
            lines.append("Also listed: " + "; ".join(others))
        st.markdown(card("Suggested facility", lines, "next"), unsafe_allow_html=True)

    if out["kind"] == "emergency":
        st.markdown('<a class="callbtn" href="tel:1122">Call 1122</a>', unsafe_allow_html=True)
        st.markdown(out["final"])
    else:
        a, h = out["advice"], HEAD[out["language"]]
        st.markdown(card(h[0], a.get("understood", "")) + card(h[1], a.get("relevant", "")), unsafe_allow_html=True)
        if a.get("to_check"):
            st.markdown(card(h[2], a["to_check"]), unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.markdown(card(h[3], a.get("do_now", [])), unsafe_allow_html=True)
        col_b.markdown(card(h[4].split(":")[0], a.get("warning_signs", []), "warn"), unsafe_allow_html=True)
        st.markdown(card(h[5], a.get("next_step", ""), "next"), unsafe_allow_html=True)
        st.download_button("Download case report for your doctor", out["report"],
                           file_name="sahaara_case_report.txt", use_container_width=True)
        with st.expander("Preview case report"):
            st.code(out["report"], language=None)
    with st.expander("Agent activity"):
        show_trace(out.get("trace"))
    queue_id = out.get("case_id") or ""
    st.markdown(f'<div class="fine">{html.escape(agents.DISCLAIMER)}'
                + (f' Queued as {html.escape(queue_id)}.' if queue_id else '')
                + '</div>', unsafe_allow_html=True)


if "result" not in st.session_state:
    st.session_state["result"] = None
if "draft" not in st.session_state:
    st.session_state["draft"] = None

if go:
    if not text.strip():
        st.warning("Please describe what you are feeling.")
        st.stop()
    measures = {k: float(v) for k, v in {"bp_sys": bp_sys, "bp_dia": bp_dia, "glucose": glucose,
                                         "temp_c": temp, "spo2": spo2}.items() if v}
    with st.spinner("Analysing your symptoms..."):
        out = graph().invoke(payload(text, age, lang, city, measures))
    if out.get("kind") == "clarify":
        st.session_state["draft"] = payload(text, age, lang, city, measures)
        st.session_state["questions"] = out.get("questions") or []
        st.session_state["pre_trace"] = out.get("trace") or []
        st.session_state["clarify_error"] = out.get("llm_error") or ""
        st.session_state["result"] = None
        st.session_state["round"] = st.session_state.get("round", 0) + 1
    else:
        st.session_state["draft"] = None
        st.session_state["result"] = out

draft = st.session_state.get("draft")
if draft and st.session_state.get("questions"):
    st.info("One or two details are missing, so guidance is paused. Answer below and the same case continues.")
    if st.session_state.get("clarify_error"):
        st.warning("Groq intake did not answer (" + st.session_state["clarify_error"] + "). These questions come from the built-in checklist.")
    with st.form("followup"):
        answers = {}
        rnd = st.session_state.get("round", 0)
        for q in st.session_state["questions"]:
            answers[q["id"]] = st.text_input(q["prompt"], key=f"fu_{rnd}_{q['id']}")
        again = st.form_submit_button("Continue", use_container_width=True)
    with st.expander("Agent activity"):
        show_trace(st.session_state.get("pre_trace"))
    if again:
        extra_lines = []
        more = {"followup_done": True}
        if (answers.get("duration") or "").strip():
            more["duration_hint"] = answers["duration"].strip()
            extra_lines.append("Duration: " + answers["duration"].strip())
        if (answers.get("city") or "").strip():
            more["city"] = answers["city"].strip()
            extra_lines.append("City: " + answers["city"].strip())
        nxt = dict(draft)
        nxt.update(more)
        if extra_lines:
            nxt["text"] = draft["text"] + "\n" + "\n".join(extra_lines)
        with st.spinner("Continuing with your answers..."):
            done = graph().invoke(nxt)
        st.session_state["draft"] = None
        st.session_state["result"] = done
        st.rerun()

if st.session_state.get("result"):
    show_result(st.session_state["result"])

with st.expander("Diagnostics"):
    st.caption(f"agents.py version: {getattr(agents, 'VERSION', 'OLD agents.py (not updated)')}")
    if st.button("Test AI connection"):
        if hasattr(agents, "diagnose"):
            res = agents.diagnose()
            (st.success if "works" in res["status"] else st.error)(res["status"])
            st.json(res)
        else:
            st.error("agents.py is the OLD version. Replace it on GitHub, then reboot the app.")
