"""Sahaara AI (simple version): a LangGraph multi-agent workflow.

Workflow:
 Intake Agent -> Triage Agent -> (RED) Emergency Agent  -> END
                              -> (else) Advice Agent -> Referral Agent (report automation) -> Safety Checker -> END
"""
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, TypedDict

from groq import Groq
from langgraph.graph import END, StateGraph

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMERGENCY = "1122 / 115"
LANG_NAMES = {"en": "simple English", "ur": "Urdu (Urdu script)", "roman_ur": "Roman Urdu (Urdu in Latin letters)"}

# ---------------------------------------------------------------- LLM helper
_client = None


def ask_json(system: str, user: str, max_tokens: int = 1200) -> dict:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            import streamlit as st
            key = st.secrets["GROQ_API_KEY"]
        _client = Groq(api_key=key)
    r = _client.chat.completions.create(
        model=MODEL, temperature=0.1, max_tokens=max_tokens, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    text = r.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}


# ---------------------------------------------------------------- small knowledge base (grounding)
KB = {
    "dizziness": "Sit/lie down at once; sip water or ORS; light snack if not eaten; stand up slowly; do not drive. Red flags: fainting, chest pain, breathlessness, face droop/arm weakness/speech trouble, head injury, confusion.",
    "weakness": "Rest, drink fluids, eat regular meals, check BP/glucose/temperature. Red flags: one-sided weakness, chest pain, breathlessness, fainting, confusion, black stools. Symptoms alone cannot confirm anemia or vitamin deficiency; testing may be needed.",
    "fainting": "If breathing: lay flat, raise legs ~30 cm if no injury, loosen clothing, fresh air; sips of water only when fully awake. Unconscious but breathing: lay on side, nothing by mouth. Every fainting episode needs a healthcare professional, ideally same day.",
    "blood_sugar": "Only if awake, thinking clearly, able to swallow AND glucose below 70 mg/dL (or low sugar strongly suspected in someone on insulin/diabetes tablets): about 15 g fast sugar (3-4 glucose tablets, half glass juice, or 1 tablespoon sugar/honey in water), recheck in 15 minutes, repeat once, then eat. Never give food/drink to a drowsy or confused person. High sugar with vomiting, stomach pain, fast breathing or drowsiness is urgent. Do not change diabetes medicines.",
    "blood_pressure": "Measure after sitting quietly 5 minutes, arm at heart level; repeat after 1-2 minutes. 180/120 or higher with chest pain, breathlessness, weakness, speech/vision change, severe headache or confusion is an emergency; without symptoms recheck, and see a clinician the same day. Low BP with dizziness: sit/lie down, fluids. Never change BP medicines.",
    "fever": "Rest, fluids, light clothing. Over-the-counter fever medicine only as per label or pharmacist. Red flags: baby under 3 months, 40 C or more, stiff neck, seizure, drowsiness, breathing trouble, non-fading rash, bleeding, persistent vomiting. See a professional if more than 3 days.",
    "dehydration": "Move to shade, small frequent sips of ORS (prepared as the packet says) or water, cool the skin. Red flags: confusion, fainting, 40 C with hot dry skin, no urine for 8 hours, cannot keep fluids down.",
    "headache": "Rest in a quiet room, drink water, eat if meals missed, check BP if possible. Red flags: sudden worst-ever headache, after head injury, fever with stiff neck, weakness/speech/vision change, confusion, pregnancy/postpartum with high BP.",
    "cough": "Sit upright, warm fluids, avoid smoke/dust. Red flags: severe breathlessness, blue lips, chest pain, coughing blood, SpO2 below 90. Cough of 2 weeks or more needs TB testing.",
    "vomiting_diarrhea": "Small frequent sips of ORS (as packet says), keep eating light food, wash hands, safe water; children: ask a health worker about zinc. Red flags: blood, severe belly pain, dehydration signs, cannot drink, infants.",
    "minor_burn": "Cool under cool running water for 20 minutes, remove rings, cover loosely with clean non-fluffy dressing; no toothpaste/butter/oil/ice; do not burst blisters. Red flags: bigger than the person's palm, face/hands/genitals/joints, chemical/electrical, deep or charred skin, children/elderly.",
    "other": "Give general, cautious advice and recommend a healthcare professional if symptoms persist or worsen.",
}

# ---------------------------------------------------------------- safety rules (work without the LLM)
RED_FLAGS = {
    "Chest pain or pressure": r"(chest (pain|tight|pressure)|seene (mein|me|main) (dard|bhari|jakar)|seena dard|dil (mein|me) dard|سینے (میں )?(درد|بھاری))",
    "Difficulty breathing": r"(saans (lene )?(mein|me|main) (bohat |boht |shadeed )?(mushkil|dikkat|takleef)|saans phool|dum ghut|can'?t breathe|difficulty breathing|short(ness)? of breath|سانس (لینے )?(میں )?(مشکل|تکلیف|پھول))",
    "Unconscious / not responding": r"(unresponsive|not responding|still unconscious|abhi (bhi )?behosh|behosh hai|jawab nahi de)",
    "Seizure": r"(seizure|convulsion|\bfits\b|daura|mirgi|مرگی|دورہ)",
    "Possible stroke signs": r"(face droop|chehra tedha|munh tedha|bolne (mein|me) (mushkil|dikkat)|slurred|ek (taraf|side) (se |ki )?(kamzori|sunn)|one side weak|منہ ٹیڑھا)",
    "Blood in vomit/stool/cough": r"(vomit(ing)? blood|khoon ki ulti|kala pakhana|black stool|khansi (mein|me) khoon|coughing blood|خون کی الٹی)",
    "Severe bleeding": r"(heavy bleeding|severe bleeding|khoon ruk nahi|bleeding won'?t stop)",
    "New confusion": r"(confus|hosh mein nahi|behki behki)",
    "Thoughts of self-harm": r"(suicide|khudkushi|kill myself|marna chahta|marna chahti|خودکشی)",
}
WATCH_FLAGS = {"Fainting / loss of consciousness": r"(behosh|bayhosh|fainted|passed out|faint|black ?out|بے ?ہوش|بیہوش)"}
NEG = re.compile(r"^[\s,.:;\-]*(?:\S+[\s,]+){0,3}?(?:nahi|nahin|nhi|not|never|no|نہیں)(?:\W|$)", re.I)


def scan(text: str, flags: Dict[str, str]) -> List[str]:
    t = text.lower()
    hits = []
    for name, pat in flags.items():
        if any(not NEG.search(t[m.end(): m.end() + 28]) for m in re.finditer(pat, t, re.I)):
            hits.append(name)
    return hits


def check_measures(m: Dict[str, float], symptomatic: bool):
    red, yellow = [], []
    g, s, d, t, sp = (m.get(k) for k in ("glucose", "bp_sys", "bp_dia", "temp_c", "spo2"))
    if g:
        if g < 54 or g >= 400:
            red.append(f"Glucose {g:.0f} mg/dL is in a dangerous range")
        elif g < 70 or g >= 300:
            yellow.append(f"Glucose {g:.0f} mg/dL is abnormal")
    if s and d:
        if s >= 180 or d >= 120:
            (red if symptomatic else yellow).append(f"BP {s:.0f}/{d:.0f} is dangerously high")
        elif s < 80 and symptomatic:
            red.append(f"BP {s:.0f}/{d:.0f} is very low with symptoms")
        elif (s < 90 or d < 60) and symptomatic:
            yellow.append(f"BP {s:.0f}/{d:.0f} is low with symptoms")
        elif s >= 140 or d >= 90:
            yellow.append(f"BP {s:.0f}/{d:.0f} is above normal")
    if t:
        if t >= 40:
            red.append(f"Temperature {t:.1f} C is very high")
        elif t >= 39.5:
            yellow.append(f"Temperature {t:.1f} C is high")
    if sp:
        if sp < 90:
            red.append(f"Oxygen saturation {sp:.0f}% is low")
        elif sp < 94:
            yellow.append(f"Oxygen saturation {sp:.0f}% is borderline")
    return red, yellow


# ---------------------------------------------------------------- static messages (3 languages)
EMERGENCY_MSG = {
    "en": "🔴 This may be an emergency. Get urgent medical help NOW: call {n} or go to the nearest hospital emergency. Do not wait for more chat. If someone is with you, ask them to help right away.",
    "roman_ur": "🔴 Yeh emergency ho sakti hai. ABHI foran tibbi madad lein: {n} par call karein ya qareeb tareen hospital ki emergency mein jayein. Is chat mein intezar na karein. Agar koi saath hai to usay abhi madad ke liye bulayein.",
    "ur": "🔴 یہ ایمرجنسی ہو سکتی ہے۔ ابھی فوری طبی مدد لیں: {n} پر کال کریں یا قریبی ہسپتال کی ایمرجنسی میں جائیں۔ اس چیٹ میں انتظار نہ کریں۔ اگر کوئی آپ کے پاس ہے تو اسے ابھی مدد کے لیے بلائیں۔",
}
FALLBACK_MSG = {
    "en": "I could not prepare guidance that is safe enough. Please see a healthcare professional today. If you have severe symptoms (breathing trouble, chest pain, fainting, confusion, seizure, heavy bleeding), call {n} now.",
    "roman_ur": "Main mehfooz rehnumai tayyar nahi kar saka. Meharbani karke aaj hi kisi healthcare professional ko dikhayein. Agar shadeed alamaat hain (saans mein dikkat, seene mein dard, behoshi, uljhan, daura, ziyada khoon) to abhi {n} par call karein.",
    "ur": "میں محفوظ رہنمائی تیار نہیں کر سکا۔ براہِ کرم آج ہی کسی ہیلتھ کیئر پروفیشنل کو دکھائیں۔ اگر شدید علامات ہیں (سانس میں دشواری، سینے میں درد، بے ہوشی، الجھن، دورہ، زیادہ خون) تو ابھی {n} پر کال کریں۔",
}
HEAD = {
    "en": ["What we understood", "What could be relevant", "What to check", "What you can do now", "Warning signs: get urgent help if", "What to do next"],
    "roman_ur": ["Hum ne kya samjha", "Kya wajah ho sakti hai", "Kya check karein", "Abhi aap kya kar sakte hain", "Khatre ki alamaat: foran madad lein agar", "Agla qadam"],
    "ur": ["ہم نے کیا سمجھا", "کیا وجہ ہو سکتی ہے", "کیا چیک کریں", "ابھی آپ کیا کر سکتے ہیں", "خطرے کی علامات: فوراً مدد لیں اگر", "اگلا قدم"],
}
BANNER = {"green": "🟢 Lower concern", "yellow": "🟡 Needs medical evaluation", "red": "🔴 Urgent"}
DISCLAIMER = "Sahaara AI gives early health information. It does not diagnose and does not replace a doctor."


# ---------------------------------------------------------------- workflow state
class State(TypedDict, total=False):
    text: str
    age_group: str
    lang_pref: str
    measures: Dict[str, float]
    language: str
    symptoms: List[str]
    duration: str
    guesses: List[str]
    topic: str
    llm_emergency: str
    needs_doctor: bool
    red: List[str]
    yellow: List[str]
    risk: str
    advice: Dict[str, Any]
    report: str
    final: str
    kind: str
    trace: List[str]


def log(s: State, msg: str) -> List[str]:
    return s.get("trace", []) + [msg]


# ---------------------------------------------------------------- agents
INTAKE_SYSTEM = f"""You are the Intake Agent of a health-support tool. The user writes English, Urdu script or Roman Urdu.
Return ONLY JSON: {{"language": "en"|"ur"|"roman_ur", "symptoms": [short English phrases], "duration": "", "guesses": [the user's own guesses such as "low sugar"],
"topic": one of {list(KB)}, "emergency_suspected": "" or a short English reason if this could be life-threatening (respect negations),
"needs_doctor": true if a professional should assess soon (symptoms >3 days, vulnerable patient, fainting, chronic illness)}}.
Never diagnose. Never invent facts."""


def intake_agent(s: State):
    try:
        d = ask_json(INTAKE_SYSTEM, f"Age group: {s['age_group']}\nMeasured values: {s['measures']}\nUser text: {s['text']}", 500)
    except Exception as e:
        d = {}
    lang = s.get("lang_pref") if s.get("lang_pref") in LANG_NAMES else d.get("language")
    topic = d.get("topic") if d.get("topic") in KB else "other"
    return {"language": lang if lang in LANG_NAMES else "roman_ur", "symptoms": d.get("symptoms") or [],
            "duration": d.get("duration") or "", "guesses": d.get("guesses") or [], "topic": topic,
            "llm_emergency": d.get("emergency_suspected") or "", "needs_doctor": bool(d.get("needs_doctor")),
            "trace": log(s, f"Intake Agent: topic={topic}, symptoms={d.get('symptoms')}")}


def triage_agent(s: State):
    red = scan(s["text"], RED_FLAGS)
    yellow = scan(s["text"], WATCH_FLAGS)
    mr, my = check_measures(s.get("measures", {}), bool(s.get("symptoms")))
    red += mr
    yellow += my
    if s.get("llm_emergency"):
        red.append(s["llm_emergency"])
    if s.get("age_group") == "newborn" and (s.get("measures", {}).get("temp_c") or 0) >= 38:
        red.append("Fever in a baby under 3 months")
    if s.get("needs_doctor") and not yellow:
        yellow.append("A healthcare professional should assess this soon")
    risk = "red" if red else "yellow" if yellow else "green"
    return {"red": red, "yellow": yellow, "risk": risk, "trace": log(s, f"Triage Agent: risk={risk}, flags={red + yellow}")}


def route_triage(s: State):
    return "emergency" if s["risk"] == "red" else "advice"


def emergency_agent(s: State):
    msg = EMERGENCY_MSG[s["language"]].format(n=EMERGENCY)
    msg += "\n\n_Reason (English): " + "; ".join(s["red"]) + "_"
    return {"final": msg, "kind": "emergency", "trace": log(s, "Emergency Agent: static emergency message (workflow interrupted)")}


ADVICE_SYSTEM = """You are the Advice Agent of a safety-first health-support tool for people with limited access to care. NEVER diagnose, NEVER prescribe.
Write every value in {lang}, using short everyday words (keep terms like BP, sugar, ORS). Return ONLY JSON with keys:
understood (1-2 sentences restating the complaint and duration), relevant (say symptoms "can be associated with several causes" and name 2-4 possibilities; the user's own guess is unconfirmed; never "you have X"),
to_check (list 0-4 useful measurements/observations), do_now (list 2-5 steps taken ONLY from GUIDELINES, keeping their conditions),
warning_signs (list 3-6 red flags from GUIDELINES), next_step (one clear sentence on who to see and when).
Rules: no medicine doses; never tell anyone to start/stop/change medicines; if risk is "yellow", say clearly a healthcare professional should evaluate; never promise the person is safe."""


def advice_agent(s: State):
    user = json.dumps({"complaint": s["text"], "age_group": s["age_group"], "symptoms": s.get("symptoms"), "duration": s.get("duration"),
                       "measured_values": s.get("measures"), "user_guesses": s.get("guesses"), "risk": s["risk"],
                       "flags": s["yellow"], "GUIDELINES": KB[s["topic"]]}, ensure_ascii=False)
    try:
        a = ask_json(ADVICE_SYSTEM.format(lang=LANG_NAMES[s["language"]]), user, 1400)
    except Exception:
        a = {}
    return {"advice": a, "trace": log(s, "Advice Agent: guidance generated from guidelines")}


def referral_agent(s: State):
    """Business-process automation: builds the referral level and a doctor-ready case report."""
    m = s.get("measures", {})
    show = lambda k, u: f"{m[k]:.4g} {u}" if m.get(k) else "not provided"
    bp = f"{m['bp_sys']:.0f}/{m['bp_dia']:.0f}" if m.get("bp_sys") and m.get("bp_dia") else "not provided"
    level = ("Visit a clinic/health centre TODAY" if s["risk"] == "yellow" else "Home care + monitoring; visit a clinic if no improvement in 48 h")
    report = "\n".join([
        "SAHAARA AI - CASE REPORT (not a diagnosis)",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Patient age group: {s['age_group']}",
        f"Main complaint: {', '.join(s.get('symptoms') or []) or s['text'][:120]}",
        f"Duration: {s.get('duration') or 'not stated'}",
        f"Measured: BP {bp} | Glucose {show('glucose', 'mg/dL')} | Temp {show('temp_c', 'C')} | SpO2 {show('spo2', '%')}",
        f"Patient's own assumption (unconfirmed): {', '.join(s.get('guesses') or []) or 'none'}",
        f"Alerts: {'; '.join(s['yellow']) or 'none'}",
        f"Triage level: {s['risk'].upper()}",
        f"Recommended referral: {level}",
    ])
    return {"report": report, "trace": log(s, f"Referral Agent: report generated, referral='{level}'")}


DOSE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|iu)\b", re.I)
DIAG = re.compile(r"\byou (definitely )?(have|are suffering from) (low|high|diabetes|anemia|anaemia|a deficiency|vitamin|iron)", re.I)
CHECK_SYSTEM = """You are a Safety Checker. Compare the ADVICE with the GUIDELINES. FAIL only if the advice: states a diagnosis as fact, gives medicine doses or tells the user to start/stop/change medicines,
adds steps not in the guidelines, or lacks warning signs / falsely reassures. Return ONLY JSON {"pass": true|false, "issues": []}."""


def safety_agent(s: State):
    a = s.get("advice") or {}
    text = json.dumps(a, ensure_ascii=False)
    issues = []
    if not a.get("warning_signs") or not a.get("next_step") or not a.get("do_now"):
        issues.append("incomplete advice")
    if DOSE.search(text):
        issues.append("medicine dose found")
    if DIAG.search(text):
        issues.append("diagnosis stated as fact")
    if not issues:
        try:
            r = ask_json(CHECK_SYSTEM, json.dumps({"GUIDELINES": KB[s["topic"]], "ADVICE": a}, ensure_ascii=False), 300)
            if r.get("pass") is not True:
                issues += r.get("issues") or ["checker did not approve"]
        except Exception as e:
            issues.append(f"checker unavailable: {e}")
    if issues:
        return {"final": FALLBACK_MSG[s["language"]].format(n=EMERGENCY), "kind": "fallback",
                "trace": log(s, f"Safety Checker: FAIL {issues} -> safe fallback")}
    h = HEAD[s["language"]]
    bullets = lambda xs: "\n".join(f"- {x}" for x in xs)
    parts = [f"**{BANNER[s['risk']]}**", f"**{h[0]}**\n{a.get('understood', '')}", f"**{h[1]}**\n{a.get('relevant', '')}"]
    if a.get("to_check"):
        parts.append(f"**{h[2]}**\n{bullets(a['to_check'])}")
    parts += [f"**{h[3]}**\n{bullets(a['do_now'])}", f"**{h[4]}**\n{bullets(a['warning_signs'])}", f"**{h[5]}**\n{a['next_step']}", f"_{DISCLAIMER}_"]
    return {"final": "\n\n".join(parts), "kind": "guidance", "trace": log(s, "Safety Checker: PASS")}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("intake", intake_agent), ("triage", triage_agent), ("emergency", emergency_agent),
                     ("advice", advice_agent), ("referral", referral_agent), ("safety", safety_agent)]:
        g.add_node(name, fn)
    g.set_entry_point("intake")
    g.add_edge("intake", "triage")
    g.add_conditional_edges("triage", route_triage, {"emergency": "emergency", "advice": "advice"})
    g.add_edge("advice", "referral")
    g.add_edge("referral", "safety")
    g.add_edge("emergency", END)
    g.add_edge("safety", END)
    return g.compile()
