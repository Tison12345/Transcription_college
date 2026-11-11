"""
Lightweight checks on the LLM's acknowledgment, and safe fallbacks.

The LLM only words a short acknowledgment. Before anything reaches the
speaker, validate_acknowledgment() rejects replies that:

  * are empty, too long, or contain role-label leftovers
  * ask a question
  * give advice or mention treatment
  * mention a body part, symptom, condition or number the patient did not
  * contradict the patient (cheerful about pain, sorry about no pain)
  * contain no acknowledgment wording at all

These are simple word rules, not a guarantee. They will not catch a
transcription error that the patient's text itself contains.
"""

import re


MAX_WORDS = 25
MAX_SENTENCES = 2

QUESTION_STARTS = (
    "what", "where", "when", "why", "how", "who", "which", "do you",
    "did you", "are you", "have you", "can you", "could you", "would you",
    "will you", "is there", "is it", "does it",
)

ADVICE_PATTERNS = [
    r"\bshould\b", r"\brecommend\w*", r"\bsuggest\w*", r"\badvi[sc]e\w*",
    r"\byou need to\b", r"\byou ought\b", r"\btry to\b", r"\bmake sure\b",
    r"\bconsider\b", r"\byou might want\b", r"\byou may want\b",
    r"\bdoctor\b", r"\bnurse\b", r"\bhospital\b", r"\bemergency\b",
    r"\bmedic\w*", r"\bpainkiller\w*", r"\bibuprofen\b", r"\bparacetamol\b",
    r"\btreat\w*", r"\bdiagnos\w*", r"\bprescri\w*", r"\bcure\b",
    r"\brest\b", r"\bice\b", r"\belevat\w*", r"\bstretch\w*",
    r"\bexercis\w*", r"\bdon't worry\b", r"\bdont worry\b",
    r"\bit will (?:get|be)\b", r"\byou will (?:get|be)\b",
]

# Words that count as "new information" if the patient did not say them.
BODY_AND_SYMPTOM_WORDS = {
    # body parts
    "foot", "feet", "ankle", "toe", "calf", "leg", "knee", "hip", "back",
    "shoulder", "arm", "hand", "wrist", "elbow", "neck", "head", "chest",
    "stomach", "heel", "thigh", "shin", "arch", "sole", "finger", "jaw",
    "abdomen", "spine", "incision", "wound", "scar",
    # symptoms / conditions
    "pain", "hurt", "ache", "sore", "swell", "swollen", "swelling",
    "discomfort", "cramp", "numb", "numbness", "tingl", "stiff", "stiffness",
    "fever", "nausea", "dizzy", "dizziness", "bleed", "bleeding", "infect",
    "infection", "fracture", "sprain", "injury", "injured", "inflammation",
    "arthritis", "headache", "bruise", "bruising", "weak", "weakness",
    "tired", "fatigue", "exhausted",
}

ACK_MARKERS = (
    "sorry", "glad", "hear", "understand", "thank", "appreciate", "noted",
    "okay", "ok", "all right", "alright", "great", "good", "wonderful",
    "happy", "sounds", "i see", "that's", "thats", "got it", "i know",
    "uncomfortable",
)

CHEERFUL = ("glad", "great", "wonderful", "happy", "good to hear", "pleased")
SYMPATHETIC = ("sorry", "sympath")


def _stem(word):
    word = word.lower()
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def _is_sensitive_word(stem_or_word):
    return any(
        stem_or_word == _stem(w) or stem_or_word == w
        for w in BODY_AND_SYMPTOM_WORDS
    )


def validate_acknowledgment(ack, patient_text, pain_reported=None):
    """
    Return (is_valid, reason). `patient_text` is what the patient said (the
    expanded yes/no meaning is fine). `pain_reported` is True / False / None.
    """

    if not ack or not ack.strip():
        return False, "empty"

    text = ack.strip()
    lowered = text.lower().replace("’", "'")

    if "\n" in text or re.search(r"\b(reply|patient|question|assistant)\s*:",
                                 lowered):
        return False, "contains role labels or multiple lines"

    words = _words(lowered)

    if len(words) > MAX_WORDS:
        return False, f"too long ({len(words)} words)"

    sentences = [s for s in re.split(r"[.!]+", text) if s.strip()]
    if len(sentences) > MAX_SENTENCES:
        return False, "too many sentences"

    if "?" in text:
        return False, "asks a question"

    if any(lowered.startswith(q + " ") for q in QUESTION_STARTS):
        return False, "phrased as a question"

    for pattern in ADVICE_PATTERNS:
        if re.search(pattern, lowered):
            return False, "gives advice or mentions treatment"

    patient_words = _words(patient_text)
    patient_stems = {_stem(w) for w in patient_words} | set(patient_words)

    for word in words:
        stem = _stem(word)
        if _is_sensitive_word(stem) or _is_sensitive_word(word):
            if stem not in patient_stems and word not in patient_stems:
                return False, f"mentions '{word}' which the patient did not"

    if any(w.isdigit() for w in words if w not in patient_words):
        return False, "introduces a number"

    if pain_reported is True and any(c in lowered for c in CHEERFUL):
        return False, "cheerful reply to a pain report"

    if pain_reported is False and any(s in lowered for s in SYMPATHETIC):
        return False, "sympathy for a patient who reported no pain"

    if not any(re.search(rf"\b{re.escape(m)}\b", lowered)
               for m in ACK_MARKERS):
        return False, "does not read as an acknowledgment"

    return True, "ok"


# Fallbacks, chosen by context. All are short, neutral and question-free.
def fallback_acknowledgment(state_name, pain_reported=None):
    if pain_reported is True:
        return "I'm sorry to hear that."
    if state_name == "INTRODUCTION":
        return "Thank you for letting me know."
    if state_name == "PAIN_LOCATION":
        return "Thank you for sharing that."
    if pain_reported is False:
        return "Thank you for letting me know."
    return "I understand."
