"""Sahaara AI (simple version): a LangGraph multi-agent workflow.

Workflow:
 Intake Agent -> Triage Agent -> (RED) Emergency Agent  -> END
                              -> (else) Advice Agent -> Referral Agent (report automation) -> Safety Checker -> END
"""
import csv
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from groq import Groq
from langgraph.graph import END, StateGraph

VERSION = "2026-10-04-v4"
# llama-3.3-70b-versatile and llama-3.1-8b-instant were shut down on 2026-08-16.
# Calling the old id is what made the deployed app fall back to built-in guidance.
PRIMARY = "openai/gpt-oss-120b"
BACKUP = "openai/gpt-oss-20b"
RETIRED = {
    "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192",
    "llama3-8b-8192", "gemma2-9b-it", "llama-3.1-70b-versatile", "mixtral-8x7b-32768",
}
EMERGENCY = "1122 / 115"
QUEUE_PATH = Path(__file__).resolve().parent / "data" / "referrals.csv"
STATUSES = ["new", "contacted", "seen", "closed"]
QUEUE_FIELDS = ["id", "created", "status", "risk", "topic", "age_group", "language",
                "city", "symptoms", "facility", "referral", "report"]
LANG_NAMES = {"en": "simple English", "ur": "Urdu (Urdu script)", "roman_ur": "Roman Urdu (Urdu in Latin letters)"}

# ---------------------------------------------------------------- LLM helper
_client = None
_last_model = PRIMARY
_models_tried: List[str] = []


def _secret(name: str) -> str:
    try:
        import streamlit as st
        val = st.secrets.get(name, "")
        if val:
            return str(val).strip()
        blob = st.secrets.get("groq", {})
        if isinstance(blob, dict):
            return str(blob.get(name, "")).strip()
    except Exception:
        return ""
    return ""


def resolve_model(explicit: str = "") -> str:
    """Pick a live Groq model. Retired ids (including the old default) map to PRIMARY."""
    chosen = (explicit or os.getenv("GROQ_MODEL") or _secret("GROQ_MODEL") or "").strip()
    if not chosen or chosen in RETIRED:
        return PRIMARY
    return chosen


def current_model() -> str:
    return _last_model


def _client_obj():
    global _client
    if _client is None:
        key = (os.getenv("GROQ_API_KEY") or "").strip() or _secret("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it under Settings → Secrets, or set the environment variable.")
        _client = Groq(api_key=key)
    return _client


def _retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(s in msg for s in ("invalid api key", "invalid_api_key", "unauthorized", "401")):
        return False
    return True


def _completion(model: str, messages: list, max_tokens: int, response_format=None, tools=None, tool_choice=None):
    params: Dict[str, Any] = {"temperature": 0.1, "max_completion_tokens": max_tokens, "messages": messages}
    if model.startswith("openai/gpt-oss") or model.startswith("qwen/"):
        params["reasoning_effort"] = "low"
    if response_format:
        params["response_format"] = response_format
    if tools:
        params["tools"] = tools
        params["tool_choice"] = tool_choice or "auto"
    def _send(body):
        return _client_obj().chat.completions.create(model=model, **body)

    try:
        return _send(params)
    except TypeError:
        params.pop("reasoning_effort", None)
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        return _send(params)
    except Exception as e:
        msg = str(e).lower()
        if "reasoning_effort" in msg or "max_completion_tokens" in msg:
            params.pop("reasoning_effort", None)
            if "max_completion_tokens" in params:
                params["max_tokens"] = params.pop("max_completion_tokens")
            return _send(params)
        raise


def model_chain() -> List[str]:
    chain = [resolve_model()]
    for name in (PRIMARY, BACKUP):
        if name not in chain:
            chain.append(name)
    return chain


def _parse_json(text: str) -> dict:
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else None
    if not isinstance(data, dict):
        raise RuntimeError("model did not return a JSON object")
    return data


def ask_json(system: str, user: str, max_tokens: int = 1200) -> dict:
    global _last_model, _models_tried
    _models_tried = []
    last: Exception = RuntimeError("no model attempted")
    for model in model_chain():
        _models_tried.append(model)
        try:
            r = _completion(
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens,
                response_format={"type": "json_object"},
            )
            text = (r.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError(f"empty response from {model}")
            _last_model = model
            return _parse_json(text)
        except Exception as e:
            last = e
            if not _retryable(e):
                break
    raise last


def diagnose() -> dict:
    """Tiny live test of the Groq connection (used by the Diagnostics panel in the UI)."""
    info = {"version": VERSION, "model": resolve_model(), "backup": BACKUP,
            "retired_blocked": "llama-3.3-70b-versatile"}
    try:
        info["reply"] = ask_json('Reply with JSON {"ok": true}', "ping", 256)
        info["model_used"] = current_model()
        info["status"] = f"AI connection works ({current_model()})"
    except Exception as e:
        info["models_tried"] = list(_models_tried)
        info["status"] = f"AI connection FAILED: {type(e).__name__}: {str(e)[:300]}"
    return info


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

TOPIC_WORDS = [
    ("fainting", r"behosh|faint|passed out|black ?out|بے ?ہوش|بیہوش"),
    ("blood_sugar", r"sugar|glucose|diabet|shugar|شوگر"),
    ("blood_pressure", r"\bbp\b|blood pressure|بی پی"),
    ("fever", r"bukhar|bukhaar|fever|بخار"),
    ("vomiting_diarrhea", r"ulti|dast|vomit|diarrh|loose motion|الٹی|دست"),
    ("cough", r"khansi|cough|کھانسی"),
    ("headache", r"sar (mein )?dard|headache|سر (میں )?درد"),
    ("minor_burn", r"jal gay|jhulas|burn|جل گ"),
    ("dehydration", r"pyaas|dehydrat|garmi|پیاس"),
    ("dizziness", r"chakkar|dizz|ghoom|چکر"),
    ("weakness", r"kamzori|weak|thakan|tired|fatigue|vitamin|khoon ki kami|anemi|anaemi|کمزوری|تھکن"),
]


def guess_topic(text: str) -> str:
    t = text.lower()
    for topic, pat in TOPIC_WORDS:
        if re.search(pat, t):
            return topic
    return "other"


def guess_lang(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "ur"
    if re.search(r"\b(mujhe|mera|meri|hai|hain|nahi|rahi|raha|aur|bohat|kya|ho)\b", text.lower()):
        return "roman_ur"
    return "en"


def standard_guidance(topic: str, risk: str, text: str) -> dict:
    """Used only when the AI service is unavailable: vetted guideline text, in English."""
    body = KB[topic]
    steps_part, _, red_part = body.partition("Red flags:")
    steps = [x.strip().rstrip(".") + "." for x in steps_part.split(". ") if x.strip()][:5]
    red = [x.strip() for x in re.split(r",| or ", red_part.strip().rstrip(".")) if x.strip()][:5]
    red += ["Symptoms getting worse, or any new serious symptom"]
    nxt = ("Please see a healthcare professional today (nearest clinic, health centre or hospital)." if risk == "yellow"
           else "Monitor at home. If it is not better within 48 hours, or any warning sign appears, see a healthcare professional.")
    return {"understood": f'You told us: "{text[:160]}"',
            "relevant": "These symptoms can be associated with several causes. Only a healthcare professional can find the cause, so please do not rely on a guess.",
            "to_check": ["Blood pressure, glucose and temperature, if you can measure them"],
            "do_now": steps, "warning_signs": red, "next_step": nxt, "degraded": True}


# ---------------------------------------------------------------- safety rules (work without the LLM)
RED_FLAGS = {
    "Chest pain or pressure": r"(chest (pain|tight|pressure)|seene (mein|me|main) (dard|bhari|jakar)|seena dard|dil (mein|me) dard|سینے (میں )?(درد|بھاری))",
    "Difficulty breathing": r"(saans (lene )?(mein|me|main) (bohat |boht |shadeed )?(mushkil|dikkat|takleef)|saans phool|saans nahi (aa|le|ar)|saans ruk|dum ghut|can'?t breathe|cannot breathe|difficulty breathing|short(ness)? of breath|سانس (لینے )?(میں )?(مشکل|تکلیف|پھول))",
    "Unconscious / not responding": r"(unresponsive|not responding|still unconscious|abhi (bhi )?behosh|behosh hai|jawab nahi de)",
    "Seizure": r"(seizure|convulsion|\bfits\b|daura|mirgi|مرگی|دورہ)",
    "Possible stroke signs": r"(face droop|chehra tedha|munh tedha|bolne (mein|me) (mushkil|dikkat)|slurred|ek (taraf|side) (se |ki )?(kamzori|sunn)|one side weak|منہ ٹیڑھا)",
    "Blood in vomit/stool/cough": r"(vomit(ing)? blood|khoon ki ulti|kala pakhana|black stool|khansi (mein|me) khoon|coughing blood|خون کی الٹی)",
    "Severe bleeding": r"(heavy bleeding|severe bleeding|khoon ruk nahi|bleeding won'?t stop)",
    "New confusion": r"(confus|hosh mein nahi|behki behki)",
    "Thoughts of self-harm": r"(suicide|khudkushi|kill myself|marna chahta|marna chahti|خودکشی)",
}
WATCH_FLAGS = {"Fainting / loss of consciousness": r"(behosh|bayhosh|fainted|passed out|faint|black ?out|بے ?ہوش|بیہوش)"}
NEG = re.compile(r"^\s*(?:\S+\s+){0,2}?(?:nahi|nahin|nhi|not|never|no|نہیں)(?:\W|$)", re.I)


NEG_BEFORE = re.compile(r"(?:\bno|\bwithout|\bbina|\bkoi|بغیر|کوئی)\s+$", re.I)
CLAUSE_END = re.compile(r"\b(aur|and|but|lekin|magar|par|bas|jabke|while)\b|[.;,!?\n]", re.I)


def scan(text: str, flags: Dict[str, str]) -> List[str]:
    """A flag is ignored only if a negation word follows it IN THE SAME CLAUSE ('chest pain nahi hai')."""
    t = text.lower()
    hits = []
    for name, pat in flags.items():
        for m in re.finditer(pat, t, re.I):
            after = t[m.end(): m.end() + 28]
            cut = CLAUSE_END.search(after)
            if cut:
                after = after[: cut.start()]
            before = t[max(0, m.start() - 10): m.start()]
            if not NEG.search(after) and not NEG_BEFORE.search(before):
                hits.append(name)
                break
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
DISCLAIMER = ("Prototype. This guidance has not been clinically reviewed. "
              "Sahaara AI does not diagnose and does not replace a doctor.")


# ---------------------------------------------------------------- facility tool + referral log
FACILITIES = [
    {"city": "karachi", "name": "Jinnah Postgraduate Medical Centre (JPMC)", "level": "tertiary emergency"},
    {"city": "karachi", "name": "Dr Ruth K. M. Pfau Civil Hospital", "level": "tertiary emergency"},
    {"city": "lahore", "name": "Mayo Hospital", "level": "tertiary emergency"},
    {"city": "lahore", "name": "Services Hospital", "level": "tertiary emergency"},
    {"city": "islamabad", "name": "Pakistan Institute of Medical Sciences (PIMS)", "level": "tertiary emergency"},
    {"city": "islamabad", "name": "Federal Government Polyclinic", "level": "hospital"},
    {"city": "rawalpindi", "name": "Benazir Bhutto Hospital", "level": "tertiary emergency"},
    {"city": "rawalpindi", "name": "Holy Family Hospital", "level": "tertiary emergency"},
    {"city": "peshawar", "name": "Lady Reading Hospital", "level": "tertiary emergency"},
    {"city": "quetta", "name": "Civil Hospital Quetta", "level": "tertiary emergency"},
    {"city": "multan", "name": "Nishtar Hospital", "level": "tertiary emergency"},
    {"city": "faisalabad", "name": "Allied Hospital", "level": "tertiary emergency"},
    {"city": "hyderabad", "name": "Liaquat University Hospital", "level": "tertiary emergency"},
]

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_facilities",
        "description": "Look up public hospitals for a city and urgency. Use this before naming a facility.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Patient city or town"},
                "risk": {"type": "string", "enum": ["green", "yellow", "red"]},
                "topic": {"type": "string"},
            },
            "required": ["city", "risk", "topic"],
        },
    },
}

PROMPTS = {
    "duration": {
        "en": "How long have you had this?",
        "roman_ur": "Yeh masla kab se hai?",
        "ur": "یہ مسئلہ کب سے ہے؟",
    },
    "city": {
        "en": "Which city are you in? A facility is looked up from your answer.",
        "roman_ur": "Aap kis sheher mein hain? Jawab se qareebi facility dhoondi jayegi.",
        "ur": "آپ کس شہر میں ہیں؟ جواب سے قریبی سہولت تلاش کی جائے گی۔",
    },
}


def lookup_facilities(city: str, risk: str, topic: str) -> dict:
    """Local directory behind the lookup_facilities tool. No invented phone numbers."""
    city_l = (city or "").strip().lower()
    hits = [f for f in FACILITIES if city_l and (f["city"] in city_l or city_l in f["city"])]
    if hits:
        note = "Public facilities from the built-in directory. Confirm the department on arrival."
    else:
        hits = [f for f in FACILITIES if "emergency" in f["level"]][:4]
        note = "That city is not in the directory. Showing major public emergency hospitals. Call 1122 for an ambulance."
    if risk == "red":
        hits = sorted(hits, key=lambda f: 0 if "emergency" in f["level"] else 1)
    hits = hits[:3]
    chosen = hits[0] if hits else {}
    return {"city": city or "", "risk": risk, "topic": topic, "matches": hits,
            "chosen": chosen.get("name", ""), "chosen_city": chosen.get("city", ""),
            "level": chosen.get("level", ""), "note": note}


def invoke_lookup(city: str, risk: str, topic: str) -> dict:
    """Force the live model to call lookup_facilities, then run that tool."""
    global _last_model
    messages = [
        {"role": "system", "content": "You are the Facility Agent. Call lookup_facilities once. Do not answer in prose."},
        {"role": "user", "content": json.dumps({"city": city, "risk": risk, "topic": topic})},
    ]
    last: Exception = RuntimeError("no model attempted")
    for model in model_chain():
        try:
            r = _completion(
                model, messages, 400, tools=[LOOKUP_TOOL],
                tool_choice={"type": "function", "function": {"name": "lookup_facilities"}},
            )
            msg = r.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                raise RuntimeError(f"{model} did not call lookup_facilities")
            args = json.loads(calls[0].function.arguments or "{}")
            found = lookup_facilities(args.get("city") or city, args.get("risk") or risk, args.get("topic") or topic)
            found["via"] = "groq"
            found["model"] = model
            _last_model = model
            return found
        except Exception as e:
            last = e
            if not _retryable(e):
                break
    raise last


def _write_queue(rows: List[dict]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=QUEUE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def log_case(row: Dict[str, str]) -> str:
    case_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    record = {k: "" for k in QUEUE_FIELDS}
    record.update({k: "" if row.get(k) is None else str(row.get(k)) for k in QUEUE_FIELDS})
    record["id"] = case_id
    record["created"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    record["status"] = "new"
    try:
        rows = list_cases()
        rows.append(record)
        _write_queue(rows)
    except Exception:
        return ""
    return case_id


def list_cases() -> List[dict]:
    if not QUEUE_PATH.exists():
        return []
    with QUEUE_PATH.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def update_status(case_id: str, status: str) -> None:
    if status not in STATUSES:
        return
    rows = list_cases()
    for row in rows:
        if row.get("id") == case_id:
            row["status"] = status
    _write_queue(rows)


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
    llm_error: str
    safety_note: str
    city: str
    duration_hint: str
    followup_done: bool
    questions: List[Dict[str, str]]
    facility: Dict[str, Any]
    case_id: str
    source: str
    model_used: str


def log(s: State, msg: str) -> List[str]:
    return s.get("trace", []) + [msg]


def guess_duration(text: str) -> str:
    m = re.search(
        r"(\d+\s*(?:day|days|din|hour|hours|ghante|ghanta|ghantay|week|weeks|hafte|hafta|haftay|month|months|mahine|mahina|minute|minutes))"
        r"|(\b(?:subah se|shaam se|kal se|aaj se|kai din|chand din|do din|teen din|since morning|since yesterday|since last night|for a few days)\b)",
        text, re.I)
    return (m.group(0).strip() if m else "")


def guess_city(text: str) -> str:
    t = text.lower()
    for city in sorted({f["city"] for f in FACILITIES}, key=len, reverse=True):
        if re.search(rf"\b{re.escape(city)}\b", t):
            return city.title()
    return ""


def missing_keys(s: State) -> List[str]:
    if s.get("followup_done") or s.get("risk") == "red":
        return []
    missing = []
    if not (s.get("duration") or "").strip():
        missing.append("duration")
    if not (s.get("city") or "").strip():
        missing.append("city")
    return missing[:2]


# ---------------------------------------------------------------- agents
INTAKE_SYSTEM = f"""You are the Intake Agent of a health-support tool. The user writes English, Urdu script or Roman Urdu.
Return ONLY JSON: {{"language": "en"|"ur"|"roman_ur", "symptoms": [short English phrases], "duration": "", "guesses": [the user's own guesses such as "low sugar"],
"topic": one of {list(KB)}, "emergency_suspected": "" or a short English reason if this could be life-threatening (respect negations),
"needs_doctor": true if a professional should assess soon (symptoms >3 days, vulnerable patient, fainting, chronic illness)}}.
Never diagnose. Never invent facts."""


def intake_agent(s: State):
    err = ""
    try:
        d = ask_json(INTAKE_SYSTEM,
                     f"Age group: {s['age_group']}\nCity: {s.get('city') or ''}\n"
                     f"Duration already given: {s.get('duration_hint') or ''}\n"
                     f"Measured values: {s['measures']}\nUser text: {s['text']}", 700)
    except Exception as e:
        d, err = {}, f"{type(e).__name__}: {str(e)[:220]}"
    lang = s.get("lang_pref") if s.get("lang_pref") in LANG_NAMES else d.get("language")
    topic = d.get("topic") if d.get("topic") in KB and d.get("topic") != "other" else guess_topic(s["text"])
    symptoms = d.get("symptoms") or ([topic.replace("_", " ")] if topic != "other" else [])
    duration = (d.get("duration") or s.get("duration_hint") or guess_duration(s["text"]) or "").strip()
    city = (s.get("city") or guess_city(s["text"]) or "").strip()
    out = {"language": lang if lang in LANG_NAMES else guess_lang(s["text"]), "symptoms": symptoms,
           "duration": duration, "city": city, "guesses": d.get("guesses") or [], "topic": topic,
           "llm_emergency": d.get("emergency_suspected") or "", "needs_doctor": bool(d.get("needs_doctor")),
           "trace": log(s, f"Intake Agent: topic={topic}, duration={duration or 'missing'}, city={city or 'missing'}"
                        + (f" (Groq error: {err})" if err else f" (Groq {current_model()})"))}
    if err:
        out["llm_error"] = err
    else:
        out["llm_error"] = ""
        out["model_used"] = current_model()
    return out


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
    if s["risk"] == "red":
        return "emergency"
    if missing_keys(s):
        return "followup"
    return "advice"


def followup_agent(s: State):
    keys = missing_keys(s)
    lang = s.get("language") if s.get("language") in LANG_NAMES else "en"
    questions = [{"id": k, "prompt": PROMPTS[k][lang]} for k in keys]
    return {"questions": questions, "kind": "clarify",
            "trace": log(s, "Follow-up Agent: asked " + ", ".join(keys))}


def _facility(s: State):
    city, risk, topic = s.get("city") or "", s.get("risk") or "yellow", s.get("topic") or "other"
    try:
        found = invoke_lookup(city, risk, topic)
        via = f"Groq tool call ({found.get('model') or current_model()})"
        err = ""
    except Exception as e:
        found = lookup_facilities(city, risk, topic)
        found["via"] = "local"
        via = "local tool"
        err = f"{type(e).__name__}: {str(e)[:160]}"
    note = f"Facility Agent: {via} lookup_facilities -> {found.get('chosen') or 'none'}"
    if err:
        note += f" [{err}]"
    return found, note


def emergency_agent(s: State):
    found, note = _facility(s)
    msg = EMERGENCY_MSG[s["language"]].format(n=EMERGENCY)
    msg += "\n\n_Reason (English): " + "; ".join(s["red"]) + "_"
    if found.get("chosen"):
        where = found["chosen"] + (f" ({found['chosen_city'].title()})" if found.get("chosen_city") else "")
        msg += f"\n\nListed emergency hospital: {where}."
    case_id = log_case({
        "risk": "red", "topic": s.get("topic", ""), "age_group": s.get("age_group", ""),
        "language": s.get("language", ""), "city": s.get("city", ""),
        "symptoms": ", ".join(s.get("symptoms") or []) or s.get("text", "")[:120],
        "facility": found.get("chosen", ""), "referral": "Emergency now", "report": msg,
    })
    return {"final": msg, "kind": "emergency", "facility": found, "case_id": case_id,
            "trace": log(s, "Emergency Agent: static emergency message; " + note)}


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
    err = ""
    try:
        a = ask_json(ADVICE_SYSTEM.format(lang=LANG_NAMES[s["language"]]), user, 1400)
    except Exception as e:
        a, err = {}, f"{type(e).__name__}: {str(e)[:220]}"
    if not (a.get("do_now") and a.get("warning_signs") and a.get("next_step")):
        a = standard_guidance(s["topic"], s["risk"], s["text"])
        note = ("Advice Agent: Groq error, draft standard guidance shown" if err
                else "Advice Agent: Groq reply was incomplete, draft standard guidance shown")
    else:
        note = f"Advice Agent: guidance generated from guidelines (Groq {current_model()})"
    out = {"advice": a, "source": "standard" if a.get("degraded") else "groq", "trace": log(s, note)}
    if not a.get("degraded"):
        out["model_used"] = current_model()
        out["llm_error"] = ""
    elif err:
        out["llm_error"] = err
    return out


def facility_agent(s: State):
    found, note = _facility(s)
    extra = {}
    if found.get("via") == "groq":
        extra["model_used"] = found.get("model") or current_model()
    return {"facility": found, "trace": log(s, note), **extra}


def referral_agent(s: State):
    """Logs the case to the referral queue and builds the handoff report."""
    m = s.get("measures", {})
    show = lambda k, u: f"{m[k]:.4g} {u}" if m.get(k) else "not provided"
    bp = f"{m['bp_sys']:.0f}/{m['bp_dia']:.0f}" if m.get("bp_sys") and m.get("bp_dia") else "not provided"
    level = ("Visit a clinic or hospital TODAY" if s["risk"] == "yellow"
             else "Home care and monitoring; visit a clinic if no improvement in 48 hours")
    fac = s.get("facility") or {}
    facility_line = fac.get("chosen") or "not matched"
    if fac.get("chosen_city"):
        facility_line += f" ({fac['chosen_city'].title()})"
    report = "\n".join([
        "SAHAARA AI — PROTOTYPE CASE REPORT",
        "Not a diagnosis. Guidance has not been clinically reviewed.",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Patient age group: {s['age_group']}",
        f"City: {s.get('city') or 'not stated'}",
        f"Main complaint: {', '.join(s.get('symptoms') or []) or s['text'][:120]}",
        f"Duration: {s.get('duration') or 'not stated'}",
        f"Measured: BP {bp} | Glucose {show('glucose', 'mg/dL')} | Temp {show('temp_c', 'C')} | SpO2 {show('spo2', '%')}",
        f"Patient's own assumption (unconfirmed): {', '.join(s.get('guesses') or []) or 'none'}",
        f"Alerts: {'; '.join(s['yellow']) or 'none'}",
        f"Triage level: {s['risk'].upper()}",
        f"Recommended referral: {level}",
        f"Facility (lookup_facilities): {facility_line}",
        f"Model: {s.get('model_used') or 'not used'}",
    ])
    case_id = log_case({
        "risk": s.get("risk", ""), "topic": s.get("topic", ""), "age_group": s.get("age_group", ""),
        "language": s.get("language", ""), "city": s.get("city", ""),
        "symptoms": ", ".join(s.get("symptoms") or []) or s.get("text", "")[:160],
        "facility": fac.get("chosen", ""), "referral": level, "report": report,
    })
    if case_id:
        report += f"\nQueue id: {case_id}"
    return {"report": report, "case_id": case_id,
            "trace": log(s, f"Referral Agent: queued {case_id or 'unlogged'}, referral='{level}'")}


DOSE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|iu)\b", re.I)
DIAG = re.compile(r"\byou (definitely )?(have|are suffering from) (low|high|diabetes|anemia|anaemia|a deficiency|vitamin|iron)", re.I)
CHECK_SYSTEM = """You are a Safety Checker. Compare the ADVICE with the GUIDELINES. FAIL only for a serious problem: the advice states a diagnosis as fact,
gives medicine doses, tells the user to start/stop/change medicines, contradicts the guidelines, or falsely reassures an urgent case. Minor wording differences, translation style or
extra general comfort measures are NOT failures. Return ONLY JSON {"pass": true|false, "issues": []}."""


def _passed(r: dict) -> bool:
    p = r.get("pass", r.get("passed", r.get("safe")))
    if isinstance(p, str):
        p = p.strip().lower() in ("true", "yes", "pass", "passed", "safe")
    return bool(p) if p is not None else not r.get("issues")


def safety_agent(s: State):
    a = s.get("advice") or {}
    if a.get("degraded"):  # vetted static text: no AI check needed, shown in English
        return {"language": "en", "kind": "guidance", "trace": log(s, "Safety Checker: skipped (vetted standard guidance)")}
    text = json.dumps(a, ensure_ascii=False)
    issues = []
    if not a.get("warning_signs") or not a.get("next_step") or not a.get("do_now"):
        issues.append("incomplete advice")
    if DOSE.search(text):
        issues.append("medicine dose found")
    if DIAG.search(text):
        issues.append("diagnosis stated as fact")
    err = s.get("llm_error", "")
    if not issues:
        try:
            r = ask_json(CHECK_SYSTEM, json.dumps({"GUIDELINES": KB[s["topic"]], "ADVICE": a}, ensure_ascii=False), 600)
            if not _passed(r):
                flagged = [str(i) for i in (r.get("issues") or ["checker did not approve"])]
                return {"kind": "guidance",
                        "safety_note": "The safety checker flagged this draft (" + "; ".join(flagged) + "). The Groq answer is still shown. It has not been clinically reviewed.",
                        "trace": log(s, f"Safety Checker: flagged {flagged}; Groq answer kept")}
        except Exception as e:
            # A failed checker must not hide a Groq answer that already passed the local rules.
            return {"kind": "guidance",
                    "safety_note": "The safety checker could not reach Groq (" + f"{type(e).__name__}: {str(e)[:160]}" + "). The advice above was not replaced.",
                    "trace": log(s, "Safety Checker: Groq checker unavailable; local checks passed, advice kept")}
    if issues:
        # Never leave the user with nothing: show the vetted standard guidance instead of the AI-written text
        std = standard_guidance(s["topic"], s["risk"], s["text"])
        return {"advice": std, "language": "en", "kind": "guidance", "source": "standard", "llm_error": err,
                "safety_note": "The AI-written answer did not pass the safety check (" + "; ".join(issues) + "), so draft standard guidance is shown instead.",
                "trace": log(s, f"Safety Checker: FAIL {issues} -> standard guidance shown")}
    return {"kind": "guidance", "model_used": current_model(), "trace": log(s, f"Safety Checker: PASS ({current_model()})")}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("intake", intake_agent), ("triage", triage_agent), ("followup", followup_agent),
                     ("emergency", emergency_agent), ("advice", advice_agent), ("facility", facility_agent),
                     ("referral", referral_agent), ("safety", safety_agent)]:
        g.add_node(name, fn)
    g.set_entry_point("intake")
    g.add_edge("intake", "triage")
    g.add_conditional_edges("triage", route_triage, {"emergency": "emergency", "followup": "followup", "advice": "advice"})
    g.add_edge("followup", END)
    g.add_edge("emergency", END)
    g.add_edge("advice", "facility")
    g.add_edge("facility", "referral")
    g.add_edge("referral", "safety")
    g.add_edge("safety", END)
    return g.compile()
