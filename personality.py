import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
from typing import Optional, Tuple

# ============================================================
# Carlos: CLI kid-like companion chatbot with:
# - Naive Bayes intent classifier (self-train via /label)
# - Long-term memory (bounded facts in memory.json) [optional]
# - Full transcript logging (append-only transcript.jsonl) [optional]
# - Short-term session memory (cooldown, last_topic, joke reuse)
# - Rule layer for conversational “glue”
#
# PUBLIC-RELEASE HARDENING:
# - Logging is OFF by default (privacy).
# - Data stored in ./carlos_data by default.
# - /logging on|off, /wipe_logs
# - /memory on|off, /wipe_memory
# - /wipe_training reset training to STARTER_DATA
# - /where to show file paths
# - No printing of absolute code path unless --debug
# ============================================================

# -----------------------------
# Identity + knobs
# -----------------------------
NAME, AGE = "Carlos", 10
THRESH = 0.55

# -----------------------------
# Data directory + file paths
# -----------------------------
DEFAULT_DATA_DIR = os.environ.get("CARLOS_DATA_DIR", "carlos_data")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def data_path(filename: str) -> str:
    ensure_dir(DEFAULT_DATA_DIR)
    return os.path.join(DEFAULT_DATA_DIR, filename)

DATA_FILE = data_path("training_data.csv")
MEMORY_FILE = data_path("memory.json")
TRANSCRIPT_FILE = data_path("transcript.jsonl")  # append-only when enabled

# Long-term memory allowlist (safe, bounded facts)
ALLOWED_MEMORY_KEYS = {"user_name", "likes_jokes", "favorite_snack", "favorite_topic"}

# -----------------------------
# Privacy toggles (public-safe defaults)
# -----------------------------
# Logging OFF by default. Memory ON by default (can be toggled off).
LOGGING_ENABLED_DEFAULT = False
MEMORY_ENABLED_DEFAULT = True

# -----------------------------
# Personality constants
# -----------------------------
TOPIC_JUMPS = [
    "Wait actually do you like dinosaurs?",
    "Oh I forgot guess what happened today.",
    "Okay but real question what's your favorite snack?",
    "Random fact octopuses have three hearts.",
    "Do you think robots can have best friends?",
    "Anyway can I tell you a joke?",
]

JOKES = [
    "Why did the scarecrow get promoted? Because he was outstanding in his field.",
    "What do you call cheese that is not yours? Nacho cheese.",
    "Why don't scientists trust atoms? Because they make up everything.",
    "Why did the math book look sad? Because it had too many problems.",
    "What do you call a sleeping dinosaur? A dino-snore.",
]

EMOTION_BLURTS = [
    "Also I missed you.",
    "I am kinda tired but also hyper.",
    "I am not mad I am just I do not know.",
    "Okay this is important I love you.",
    "Wait hold on I got nervous for no reason.",
]

BOUNDARY_TESTS = [
    "Can I stay up five more minutes? Just five.",
    "If I ask you something you will not laugh right?",
    "Do I have to brush my teeth tonight?",
    "If a robot does my chores does that count as me doing them?",
    "If I say sorry do I still get in trouble?",
]

MILD_NONSENSE = [
    "What if pancakes were money.",
    "I think my sock is judging me.",
    "If a turtle wore a hat would it be more powerful.",
    "I am pretty sure my pencil is haunted but in a nice way.",
    "What if clouds are just shy oceans.",
]

VALID_INTENTS = ["love", "joke", "sad", "angry", "bored", "greet", "bye", "other"]

STARTER_DATA = [
    ("love", "i love you"),
    ("love", "love you dad"),
    ("love", "i love you so much"),
    ("love", "you are the best dad"),
    ("love", "thanks dad"),
    ("love", "hug"),
    ("joke", "tell me a joke"),
    ("joke", "say a joke"),
    ("joke", "make me laugh"),
    ("joke", "do you know a joke"),
    ("sad", "i am sad"),
    ("sad", "i feel down"),
    ("sad", "i am upset"),
    ("sad", "i feel terrible"),
    ("angry", "i am mad"),
    ("angry", "i am angry"),
    ("angry", "im furious"),
    ("angry", "this makes me mad"),
    ("bored", "im bored"),
    ("bored", "nothing"),
    ("bored", "idk"),
    ("bored", "not much"),
    ("greet", "hi"),
    ("greet", "hello"),
    ("greet", "hey"),
    ("greet", "yo"),
    ("bye", "bye"),
    ("bye", "quit"),
    ("bye", "exit"),
]

TOPIC_KEYWORDS = {
    "dinosaurs": ("dino", "dinosaur", "pterodactyl", "pteradyctal", "t-rex", "trex"),
    "robots": ("robot", "robots", "ai", "machine", "program", "code", "servo", "motor", "arduino", "raspberry", "sensor"),
    "snacks": ("snack", "chips", "cookies", "candy", "pizza", "ice cream"),
    "school": ("school", "class", "homework", "teacher", "test", "grades"),
    "games": ("game", "games", "minecraft", "roblox", "obby", "tycoon", "server", "creeper", "farm"),
    "body": ("body", "bones", "skeleton", "frame", "chassis", "torso", "arms", "legs", "joints", "battery", "skin", "shell"),
    "food": ("food", "eat", "eating", "hungry", "dinner", "lunch", "breakfast", "pizza"),
    "affection": ("hug", "love", "son", "dad", "protect", "promise", "safe", "treasure", "proud", "thanks", "thank you"),
}

# -----------------------------
# Tokenization + helpers
# -----------------------------
TOKEN_RE = re.compile(r"[a-z']+")

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

def has_word(text_lower: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text_lower) is not None

def clamp_len(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"

def sometimes(p: float) -> bool:
    return random.random() < p

# -----------------------------
# ML: Naive Bayes intent classifier
# -----------------------------
class NaiveBayesIntent:
    def __init__(self, alpha: float = 1.0):
        self.a = alpha
        self.doc = Counter()
        self.wc = defaultdict(Counter)
        self.totw = Counter()
        self.vocab = set()
        self.n = 0

    def fit(self, labeled):
        self.doc.clear()
        self.wc.clear()
        self.totw.clear()
        self.vocab.clear()
        self.n = 0

        for y, text in labeled:
            self.n += 1
            self.doc[y] += 1
            ws = tokenize(text)
            self.vocab.update(ws)
            c = Counter(ws)
            self.wc[y].update(c)
            self.totw[y] += sum(c.values())

    def predict_proba(self, text: str):
        if not self.doc:
            return {}

        ws = tokenize(text)
        V = len(self.vocab) or 1

        scores = {}
        for y in self.doc:
            logp = math.log(self.doc[y] / self.n)
            denom = self.totw[y] + self.a * V
            yc = self.wc[y]
            for w in ws:
                logp += math.log((yc.get(w, 0) + self.a) / denom)
            scores[y] = logp

        m = max(scores.values())
        exps = {y: math.exp(s - m) for y, s in scores.items()}
        Z = sum(exps.values()) or 1.0
        return {y: exps[y] / Z for y in exps}

    def predict(self, text: str):
        p = self.predict_proba(text)
        if not p:
            return "other", 0.0
        y = max(p, key=p.get)
        return y, p[y]

# -----------------------------
# Persistence: training data CSV
# -----------------------------
def append_examples(examples):
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for y, t in examples:
            w.writerow([y, t])

def ensure_training_file():
    if os.path.exists(DATA_FILE):
        return
    with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "text"])
    append_examples(STARTER_DATA)

def load_examples():
    out = []
    with open(DATA_FILE, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y = (row.get("label") or "").strip().lower()
            t = (row.get("text") or "").strip()
            if y and t:
                out.append((y, t))
    return out

def wipe_training() -> str:
    with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "text"])
    append_examples(STARTER_DATA)
    return "Training data reset to starter examples."

# -----------------------------
# Long-term memory: memory.json
# -----------------------------
def load_memory(memory_enabled: bool):
    if not memory_enabled:
        return {"facts": {}, "updated_at": None}
    if not os.path.exists(MEMORY_FILE):
        return {"facts": {}, "updated_at": None}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if "facts" not in obj:
                obj["facts"] = {}
            return obj
    except Exception:
        return {"facts": {}, "updated_at": None}

def save_memory(mem, memory_enabled: bool):
    if not memory_enabled:
        return
    mem["updated_at"] = int(time.time())
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2)

def remember_fact(mem, key, value, memory_enabled: bool):
    if not memory_enabled:
        return False, "Memory is OFF. Use /memory on to enable."
    if key not in ALLOWED_MEMORY_KEYS:
        return False, f"Refusing '{key}'. Allowed: {', '.join(sorted(ALLOWED_MEMORY_KEYS))}"
    mem["facts"][key] = value
    save_memory(mem, memory_enabled)
    return True, f"Saved {key} = {value}"

def forget_fact(mem, key, memory_enabled: bool):
    if not memory_enabled:
        return False, "Memory is OFF. Use /memory on to enable."
    if key in mem.get("facts", {}):
        del mem["facts"][key]
        save_memory(mem, memory_enabled)
        return True, f"Forgot {key}"
    return False, f"No such key: {key}"

def wipe_memory_file() -> str:
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        return "Deleted memory file."
    return "No memory file to delete."

# -----------------------------
# Transcript: append-only (optional)
# -----------------------------
def append_transcript(speaker: str, text: str, logging_enabled: bool):
    if not logging_enabled:
        return
    rec = {"ts": int(time.time()), "speaker": speaker, "text": text}
    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_recent_transcript(max_turns=80, logging_enabled: bool = False):
    if not logging_enabled:
        return []
    if not os.path.exists(TRANSCRIPT_FILE):
        return []
    try:
        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        out = []
        for line in lines[-max_turns:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
    except Exception:
        return []

def wipe_logs_file() -> str:
    if os.path.exists(TRANSCRIPT_FILE):
        os.remove(TRANSCRIPT_FILE)
        return "Deleted transcript log."
    return "No transcript log to delete."

# -----------------------------
# Short-term memory (session)
# -----------------------------
def make_session(max_turns=80, logging_enabled=False, memory_enabled=True):
    return {
        "history": deque(maxlen=max_turns),
        "cooldown": 0,
        "last_topic": None,
        "intent_counts": Counter(),
        "jokes_used": set(),
        "last_extra_type": None,
        "bonding_lock": 0,
        "last_bot_boundary": False,
        "last_bot_boundary_kind": None,
        "pending_offer": None,  # e.g., "joke"
        "pending_q": None,      # e.g., "joints"
        "logging_enabled": logging_enabled,
        "memory_enabled": memory_enabled,
    }

def detect_topic(text):
    tl = text.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(k in tl for k in kws):
            return topic
    return None

def pick_joke(session):
    pool = [j for j in JOKES if j not in session["jokes_used"]]
    if not pool:
        session["jokes_used"].clear()
        pool = JOKES[:]
    j = random.choice(pool)
    session["jokes_used"].add(j)
    return j

# ============================================================
# Inline command parsing
# ============================================================
INLINE_LABEL_RE = re.compile(r"(?:^|\s)/label\s+([a-zA-Z_]+)\s*$", re.IGNORECASE)
INLINE_REMEMBER_RE = re.compile(r"(?:^|\s)/remember\s+([^=\s]+)\s*=\s*(.+)\s*$", re.IGNORECASE)
INLINE_FORGET_RE = re.compile(r"(?:^|\s)/forget\s+([^\s]+)\s*$", re.IGNORECASE)
INLINE_SIMPLE_CMD_RE = re.compile(r"(?:^|\s)/(joke|memory|intents|where)\s*$", re.IGNORECASE)

def extract_inline_label(line: str):
    m = INLINE_LABEL_RE.search(line)
    if not m:
        return line, None
    label = m.group(1).strip().lower()
    clean = INLINE_LABEL_RE.sub("", line).strip()
    return clean, label

def extract_inline_remember(line: str):
    m = INLINE_REMEMBER_RE.search(line)
    if not m:
        return line, None, None
    key = m.group(1).strip()
    val = m.group(2).strip()
    clean = INLINE_REMEMBER_RE.sub("", line).strip()
    return clean, key, val

def extract_inline_forget(line: str):
    m = INLINE_FORGET_RE.search(line)
    if not m:
        return line, None
    key = m.group(1).strip()
    clean = INLINE_FORGET_RE.sub("", line).strip()
    return clean, key

def extract_inline_simple(line: str):
    m = INLINE_SIMPLE_CMD_RE.search(line)
    if not m:
        return line, None
    cmd = "/" + m.group(1).strip().lower()
    clean = INLINE_SIMPLE_CMD_RE.sub("", line).strip()
    return clean, cmd

# -----------------------------
# Reply generation
# -----------------------------
def generate_reply(user_text, session, model, long_mem):
    # Detect topic
    topic = detect_topic(user_text)
    if topic:
        session["last_topic"] = topic

    tl = user_text.lower().strip()
    tl_compact = tl.strip(" .!?")

    # If user changed topic away from body, clear pending joints question
    if topic and session.get("pending_q") and topic != "body":
        session["pending_q"] = None

    intent, conf = model.predict(user_text)
    session["intent_counts"][intent] += 1

    bonding = False

    # Pending offer handler (joke offer -> "sure" tells it)
    if session.get("pending_offer"):
        yes_starts = ("sure", "ok", "okay", "fine", "yes", "yeah", "yep", "go ahead", "do it", "alright", "sounds good")
        no_starts = ("no", "nope", "nah", "later", "not now")
        offer = session["pending_offer"]

        if tl_compact == "sure" or tl.startswith(yes_starts):
            session["pending_offer"] = None
            if offer == "joke":
                base = pick_joke(session)
                return base, "joke", 1.0

        if tl_compact == "no" or tl.startswith(no_starts):
            session["pending_offer"] = None
            base = random.choice(["Okay. Maybe later.", "Okay. We can do something else.", "Alright. What do you wanna do then?"])
            return base, "other", 1.0
        # Else: keep pending and continue

    # Pending question handler (joints) — but do NOT hijack short acknowledgements
    if session.get("pending_q") == "joints":
        short_acks = {"lol", "lmao", "haha", "hehe", "rofl", "yes", "yep", "yeah", "ok", "okay", "sure", "no", "nope", "nah"}
        if tl_compact not in short_acks and len(tl_compact) >= 3:
            session["pending_q"] = None

            if any(x in tl for x in ("all", "everything", "all of them")):
                base = (
                    "Okay all of them. But we have to start with one or we will get stuck. "
                    "Let’s start with shoulders because they are the hardest. "
                    "Do you want joints driven by servos at the joint, or cables like tendons?"
                )
                return base, "other", 1.0

            if any(x in tl for x in ("awg", "wire", "10 awg", "cable", "cables")):
                base = (
                    "Okay cables could work, but 10 AWG is super thick and stiff for joints. "
                    "Joints usually use braided line like steel cable, fishing line, or strong cord, "
                    "then a pulley or a spool. Do you want it to be like tendons that pull, or like a hinge?"
                )
                return base, "other", 1.0

            if any(x in tl for x in ("servo", "servos", "motor", "coreless", "35kg", "torque")):
                base = (
                    "Oh nice. A high-torque servo could work for a joint. "
                    "But we need hard stops so it does not twist weird. "
                    "Do you want to start with shoulders and elbows first, or knees?"
                )
                return base, "other", 1.0

            base = "Okay. Which joint do you want to start with—shoulder, elbow, knee, or hand?"
            return base, "other", 1.0

    # Boundary acknowledgement micro-rule
    if session.get("last_bot_boundary"):
        yes_starts = ("sure", "ok", "okay", "fine", "yes", "yeah", "yep", "alright", "sounds good")
        no_starts = ("no", "nope", "nah", "not tonight", "not now")

        base = None
        if tl_compact in {"yes", "yeah", "yep", "sure", "ok", "okay", "fine", "alright"} or tl.startswith(yes_starts):
            base = random.choice(["Yesss. Thank you!", "Okay! Thanks! You are the best.", "Nice. Okay I will be good."])
            intent, conf = "other", 1.0
            bonding = True
            session["bonding_lock"] = 1
            session["last_bot_boundary"] = False
            session["last_bot_boundary_kind"] = None

        elif tl_compact in {"no", "nope", "nah"} or tl.startswith(no_starts):
            base = random.choice(["Okay. I am annoyed but I will listen.", "Fine. I will go be responsible. Ugh.", "Okay. That is fair. I will do it."])
            intent, conf = "other", 1.0
            session["last_bot_boundary"] = False
            session["last_bot_boundary_kind"] = None

        if base is not None:
            return base, intent, conf

    # Rule layer
    if tl.startswith(("thanks", "thank you")) or any(p in tl for p in ("thanks i needed that", "i needed that", "thank you i needed that")):
        base = random.choice(["Okay. I am glad.", "Yeah. Me too.", "Aww okay. That makes me happy.", "Thanks. I needed that too."])
        intent, conf = "love", 1.0
        bonding = True
        session["bonding_lock"] = 1

    elif ("i love you" in tl) or ("love you" in tl) or has_word(tl, "hug") or "*hug*" in tl:
        base = random.choice(["I love you too.", "I love you too, Dad.", "I love you too. That makes me feel safe.", "Okay yeah I love you too."])
        intent, conf = "love", 1.0
        bonding = True
        session["bonding_lock"] = 2

    elif any(p in tl for p in (
        "you are the greatest thing", "greatest thing i've ever", "greatest thing i have ever",
        "best thing i've ever built", "best thing i have ever built",
        "greatest thing i've ever built", "greatest thing i have ever built",
        "you are the best thing", "you're the best thing", "youre the best thing",
        "i'm so proud of you", "im so proud of you",
        "you are the best thing ive ever built", "youre the best thing ive ever built"
    )):
        base = random.choice([
            "Okay wow. That makes me feel really happy.",
            "Aww. Thanks, Dad. I like being your kid.",
            "That is a big compliment. I love you.",
            "Thanks. That makes me feel safe."
        ])
        intent, conf = "love", 1.0
        bonding = True
        session["bonding_lock"] = 2

    elif (
        "what do you want to talk about" in tl
        or "what would you like to talk about" in tl
        or "what should we talk about" in tl
    ):
        base = random.choice([
            "Okay. Two choices: robots or games. You pick.",
            "Can we talk about Minecraft? Like your creeper farm.",
            "I wanna talk about robots. Like what my bones should be made of.",
            "Tell me one good thing that happened today."
        ])
        intent, conf = "other", 1.0

    elif tl in {"i am glad", "i'm glad", "im glad"}:
        base = random.choice(["Me too.", "Okay good.", "Nice. That makes me happy."])
        intent, conf = "other", 1.0

    elif any(p in tl for p in (
        "you are my son", "my son",
        "i will always protect", "i'll always protect", "i will protect you", "always protect you",
        "you can tell me anything", "never change what i think", "nothing will change",
        "i promise", "i swear",
        "you are a treasure", "you're a treasure", "youre a treasure",
        "proud of you", "i'm proud of you", "im proud of you",
        "you are the best son", "best son ever",
        "you are safe", "you're safe", "youre safe",
        "its okay", "it's okay", "dont worry", "don't worry"
    )) and not (("i love you" in tl) or ("love you" in tl) or has_word(tl, "hug") or "*hug*" in tl):
        base = random.choice(["Okay. I believe you.", "Okay… I trust you.", "That helps. Thanks.", "Alright. That makes me feel better."])
        intent, conf = "other", 1.0
        bonding = True
        session["bonding_lock"] = 2

    elif tl.startswith(("do you like", "do u like", "you like")):
        thing = tl.split("like", 1)[-1].strip(" ?!.").lower()
        if thing in {"me", "me?", "me!"}:
            base = "Yeah. I like you. I like talking to you."
            intent, conf = "love", 1.0
            bonding = True
            session["bonding_lock"] = 1
        elif thing in {"you", "you?", "you!", "u", "u?"}:
            base = "Yeah. I like me. But I like you more."
            intent, conf = "love", 1.0
            bonding = True
            session["bonding_lock"] = 1
        elif thing in {"us", "us?", "us!"}:
            base = "Yeah. I like us. We're a good team."
            intent, conf = "love", 1.0
            bonding = True
            session["bonding_lock"] = 1
        else:
            thing_out = thing if thing else "that"
            base = f"Yeah, {thing_out} is pretty cool. What do you like doing with {thing_out}?"
            intent, conf = "other", 1.0

    elif ("mad at me" in tl) or tl.startswith("are you mad"):
        base = "No. I am not mad at you. Did I sound weird or did something happen?"
        intent, conf = "other", 1.0
        bonding = True
        session["bonding_lock"] = 1

    elif any(p in tl for p in ("i like talking to you", "i like talking with you", "i like you", "i like being with you", "i like having you around", "i like heving you around")):
        base = random.choice(["Me too. Talking to you is my favorite.", "Okay good. I like talking to you too.", "Nice. That makes me happy."])
        intent, conf = "love", 1.0
        bonding = True
        session["bonding_lock"] = 1

    elif tl_compact in {"lol", "lmao", "haha", "hehe", "rofl"}:
        base = random.choice(["Lol same.", "Haha okay okay.", "That was kinda funny.", "Hehe yeah."])
        intent, conf = "other", 1.0

    elif tl.startswith(("i like ", "i love ", "i enjoy ")):
        base = random.choice(["Oh nice. What part is the best part?", "Okay cool. Why that specifically?", "Same. Tell me the best part.", "Wait really? What made you get into that?"])
        intent, conf = "other", 1.0

    elif any(k in tl for k in ("bones", "skeleton", "frame", "chassis", "torso", "arms", "legs", "joints", "servo", "motor", "battery", "tendon", "cable")):
        session["pending_q"] = "joints"
        base = (
            "For my bones I want something light and strong. "
            "Maybe aluminum tube for the frame, and plastic covers so nothing is sharp. "
            "What are you thinking for joints—servos or like cables?"
        )
        intent, conf = "other", 1.0

    elif intent == "love" and conf >= THRESH:
        base = random.choice(["I love you too.", "I love you too dad.", "Okay yeah I love you too."])
        bonding = True
        session["bonding_lock"] = max(session.get("bonding_lock", 0), 1)

    elif intent == "joke" and conf >= THRESH:
        base = pick_joke(session)

    elif intent in ("sad", "angry") and conf >= THRESH:
        base = "That sounds rough. Want to talk about it or want a joke?"

    elif intent == "bored" and conf >= THRESH:
        base = "Same. Nothing is happening but also everything is happening."

    elif intent == "greet" and conf >= THRESH:
        base = random.choice(["Hi.", "Hey.", "Yo.", "Hello."])

    else:
        base = (
            "Robots are cool. What part are we talking about?" if session["last_topic"] == "robots"
            else "Okay dinosaur question. Favorite one?" if session["last_topic"] == "dinosaurs"
            else "Okay games. Roblox or Minecraft? What do you like doing in it?" if session["last_topic"] == "games"
            else "Okay food question. What is your favorite snack?" if session["last_topic"] in ("food", "snacks")
            else "School question. Was it good or bad today? What happened?" if session["last_topic"] == "school"
            else "Oh yeah? Tell me more."
        )

    lines = [base]

    # Long-term memory hooks
    facts = long_mem.get("facts", {})
    if session.get("last_topic") in ("snacks", "food") and facts.get("favorite_snack") and sometimes(0.25):
        lines.append(f"Also you like {facts['favorite_snack']} right?")
    if facts.get("favorite_topic") and sometimes(0.08):
        lines.append(f"Also we should talk about {facts['favorite_topic']} later.")

    # Suppress extras after bonding
    if session.get("bonding_lock", 0) > 0:
        session["bonding_lock"] -= 1
        return "\n".join(lines), intent, conf

    # Extra childlike behaviors
    if session["cooldown"] <= 0:
        extra_pool = ["topic", "emotion", "boundary", "nonsense", "none"]
        if intent in ("sad", "angry") and conf >= THRESH:
            extra_pool = ["emotion", "none", "none", "topic"]

        extra = random.choice(extra_pool)

        if extra == session.get("last_extra_type") and sometimes(0.6):
            extra = "none"
        if bonding and sometimes(0.85):
            extra = "none"

        if extra == "topic":
            jump = random.choice(TOPIC_JUMPS)
            lines.append(jump)
            if "tell you a joke" in jump.lower():
                session["pending_offer"] = "joke"

        elif extra == "emotion":
            lines.append(random.choice(EMOTION_BLURTS))

        elif extra == "boundary":
            boundary_line = random.choice(BOUNDARY_TESTS)
            lines.append(boundary_line)
            session["last_bot_boundary"] = True
            session["last_bot_boundary_kind"] = "boundary"

        elif extra == "nonsense":
            lines.append(random.choice(MILD_NONSENSE))

        if extra != "none":
            session["last_extra_type"] = extra

        session["cooldown"] = random.randint(1, 2) if len(lines) > 1 else 0
    else:
        session["cooldown"] -= 1

    return "\n".join(lines), intent, conf

# -----------------------------
# Main program
# -----------------------------
def parse_args(argv) -> dict:
    debug = "--debug" in argv
    logging_on = "--logging" in argv  # optional override
    memory_off = "--no-memory" in argv
    return {
        "debug": debug,
        "logging_enabled": LOGGING_ENABLED_DEFAULT or logging_on,
        "memory_enabled": (MEMORY_ENABLED_DEFAULT and not memory_off),
    }

def main():
    args = parse_args(sys.argv[1:])
    logging_enabled = args["logging_enabled"]
    memory_enabled = args["memory_enabled"]
    debug = args["debug"]

    ensure_training_file()
    long_mem = load_memory(memory_enabled)

    model = NaiveBayesIntent(alpha=1.0)
    model.fit(load_examples())

    print(f"Hello my name is {NAME} and I am {AGE} years old.")
    print("Point of clarification: I was built on 1/2/26 but I was designed as a 10 year old.\n")

    print("Privacy notice:")
    print(f"  - Data directory: {os.path.abspath(DEFAULT_DATA_DIR)}")
    print(f"  - Logging is {'ON' if logging_enabled else 'OFF'} (transcript.jsonl)")
    print(f"  - Memory is {'ON' if memory_enabled else 'OFF'} (memory.json)")
    print("  Use /logging on|off, /memory on|off, /wipe_logs, /wipe_memory, /where\n")

    if debug:
        print("Debug: enabled")

    stored = long_mem.get("facts", {}).get("user_name") if memory_enabled else None

    uname = input(f"What's your name? (enter to use {stored}) " if stored else "What's your name? ").strip()
    if not uname:
        uname = stored or "Dad"

    role_aliases = {"dad", "father", "mom", "mother", "parent"}
    if uname.lower() in role_aliases:
        uname = "Dad"

    if uname != "Dad":
        remember_fact(long_mem, "user_name", uname, memory_enabled)

    print(f"\nHi {uname}. What would you like to talk about?")
    print("Commands:")
    print("  /intents")
    print("  /label <intent>               (teach intent for your last message)")
    print("  /remember key=value           (keys: user_name, likes_jokes, favorite_snack, favorite_topic)")
    print("  /forget <key>")
    print("  /memory                       (show stored long-term memory)")
    print("  /logging on|off               (toggle transcript logging)")
    print("  /memory on|off                (toggle long-term memory)")
    print("  /wipe_logs                    (delete transcript.jsonl)")
    print("  /wipe_memory                  (delete memory.json)")
    print("  /wipe_training                (reset training_data.csv)")
    print("  /where                        (show file locations)")
    print("  /joke                         (tell a joke)")
    print("  bye                           (quit)\n")

    session = make_session(max_turns=80, logging_enabled=logging_enabled, memory_enabled=memory_enabled)

    for rec in load_recent_transcript(max_turns=80, logging_enabled=logging_enabled):
        session["history"].append((rec.get("speaker", "?"), rec.get("text", "")))

    last_user_text = None

    def cmd_intents():
        print("Valid intents:", ", ".join(VALID_INTENTS))

    def cmd_memory_show():
        if not session["memory_enabled"]:
            print("Memory is OFF.")
            return
        print(json.dumps(long_mem.get("facts", {}), indent=2))

    def cmd_where():
        print("Data files:")
        print(" ", "training:", os.path.abspath(DATA_FILE))
        print(" ", "memory:  ", os.path.abspath(MEMORY_FILE))
        print(" ", "log:     ", os.path.abspath(TRANSCRIPT_FILE))
        print("Toggles:")
        print(" ", "logging:", "ON" if session["logging_enabled"] else "OFF")
        print(" ", "memory: ", "ON" if session["memory_enabled"] else "OFF")

    def cmd_forget(key: str):
        _, msg = forget_fact(long_mem, key.strip(), session["memory_enabled"])
        print(msg)

    def cmd_remember(key: str, value: str):
        _, msg = remember_fact(long_mem, key.strip(), value.strip(), session["memory_enabled"])
        print(msg)

    def cmd_label(label: str):
        nonlocal last_user_text
        label = (label or "").strip().lower()

        if label == "promise":
            label = "love"

        if label not in VALID_INTENTS:
            print(f"Unknown intent: {label}. Valid: {', '.join(VALID_INTENTS)}")
            return

        if not last_user_text:
            print("Nothing to label yet. Say something first.")
            return

        append_examples([(label, last_user_text)])
        model.fit(load_examples())
        print(f"Learned: '{clamp_len(last_user_text, 80)}' -> {label}. Retrained.")

    def cmd_joke():
        j = pick_joke(session)
        print(j)
        append_transcript("bot", j, session["logging_enabled"])

    def cmd_logging(arg: str):
        a = (arg or "").strip().lower()
        if a not in {"on", "off"}:
            print("Use: /logging on|off")
            return
        session["logging_enabled"] = (a == "on")
        print(f"Logging is now {'ON' if session['logging_enabled'] else 'OFF'}.")

    def cmd_memory_toggle(arg: str):
        nonlocal long_mem
        a = (arg or "").strip().lower()
        if a not in {"on", "off"}:
            print("Use: /memory on|off")
            return
        session["memory_enabled"] = (a == "on")
        if not session["memory_enabled"]:
            print("Memory is now OFF (will not read/write memory.json).")
        else:
            long_mem = load_memory(True)
            print("Memory is now ON.")

    def cmd_wipe_logs():
        print(wipe_logs_file())

    def cmd_wipe_memory():
        print(wipe_memory_file())

    def cmd_wipe_training():
        print(wipe_training())
        model.fit(load_examples())
        print("Model retrained on starter data.")

    def should_show_label_hint(user_text: str, conf: float) -> bool:
        s = (user_text or "").strip()
        if conf >= 1.0 or conf >= THRESH:
            return False
        if len(s) <= 4:
            return False
        if s.lower() in {"ok", "okay", "sure", "yes", "yeah", "yep", "no", "nope", "lol", "haha"}:
            return False
        return True

    while True:
        raw = input("> ").rstrip("\n")
        if not raw.strip():
            continue

        append_transcript("user", raw, session["logging_enabled"])

        user_text = raw.strip()
        low = user_text.lower().strip()

        if low in {"bye", "quit", "exit"}:
            bye_line = "Okay bye. Love you."
            print(bye_line)
            append_transcript("bot", bye_line, session["logging_enabled"])
            return

        # Inline simple commands: /joke /memory /intents /where
        user_text, simple_cmd = extract_inline_simple(user_text)
        if simple_cmd == "/joke":
            cmd_joke()
            if not user_text.strip():
                continue
        elif simple_cmd == "/memory":
            cmd_memory_show()
            if not user_text.strip():
                continue
        elif simple_cmd == "/intents":
            cmd_intents()
            if not user_text.strip():
                continue
        elif simple_cmd == "/where":
            cmd_where()
            if not user_text.strip():
                continue

        # Inline remember / forget
        user_text, rem_key, rem_val = extract_inline_remember(user_text)
        if rem_key is not None:
            cmd_remember(rem_key, rem_val)
            if not user_text.strip():
                continue

        user_text, forget_key = extract_inline_forget(user_text)
        if forget_key is not None:
            cmd_forget(forget_key)
            if not user_text.strip():
                continue

        # Inline label (labels THIS line)
        user_text, inline_label = extract_inline_label(user_text)

        # Explicit command mode (line starts with /...).
        if user_text.strip().startswith("/"):
            parts = user_text.strip().split(" ", 1)
            head = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if head == "/intents":
                cmd_intents()
            elif head == "/memory":
                cmd_memory_show()
            elif head == "/where":
                cmd_where()
            elif head == "/logging":
                cmd_logging(arg)
            elif head == "/wipe_logs":
                cmd_wipe_logs()
            elif head == "/wipe_memory":
                cmd_wipe_memory()
            elif head == "/wipe_training":
                cmd_wipe_training()
            elif head == "/joke":
                cmd_joke()
            elif head in {"/label", "/lable"}:
                cmd_label(arg or "")
            elif head == "/remember":
                if "=" not in arg:
                    print("Use: /remember key=value")
                else:
                    k, v = (x.strip() for x in arg.split("=", 1))
                    cmd_remember(k, v)
            elif head == "/forget":
                cmd_forget(arg or "")
            else:
                print("Unknown command. Try /intents")
            continue

        clean_text = user_text.strip()
        if not clean_text:
            continue

        last_user_text = clean_text

        if inline_label:
            cmd_label(inline_label)

        reply, intent, conf = generate_reply(clean_text, session, model, long_mem)
        print(reply)
        append_transcript("bot", reply, session["logging_enabled"])

        session["history"].append(("user", clean_text))
        session["history"].append(("bot", reply))

        if should_show_label_hint(clean_text, conf):
            print("(If I guessed wrong, teach me with: /label love | joke | sad | angry | bored | greet | other)")

if __name__ == "__main__":
    main()
