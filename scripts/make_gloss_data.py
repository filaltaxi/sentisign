"""Synthetic gloss -> English pairs for fine-tuning Flan-T5 on WLASL300 vocabulary.
ASL-style gloss order (time first, topic-comment, WH-word last, NO before verb, FINISH after verb) mapped to
natural English. Emotion changes punctuation only; the voice carries the feeling, and the meaning never changes.
Templated data: the model learns these patterns; a human-written test set is still needed for a fair score.
Writes data/gloss_en_train.jsonl, data/gloss_en_test.jsonl (held-out word combinations), data/gloss_en_review.csv (300 to hand-check)."""
import json, random, csv, os, argparse

FOOD = "apple banana candy cereal cheese coffee corn egg fish meat milk pizza salt sandwich tea tomato water".split()
DRINK = {"coffee", "milk", "tea", "water"}
PEOPLE = "baby boy brother children cousin daughter father friend girl husband man mother son student teacher wife woman doctor police secretary family".split()
RELATIVE = {"brother", "cousin", "daughter", "father", "friend", "husband", "mother", "son", "wife", "family"}
PLACES = {"bathroom": "the bathroom", "city": "the city", "class": "class", "college": "college", "home": "home", "house": "the house",
          "office": "the office", "restaurant": "the restaurant", "room": "my room", "school": "school"}
THINGS = "backpack ball bed blanket book chair clock clothes computer dress glasses hat jacket knife letter list money pants paper pencil shirt table window".split()
ADJ = {"bad": "bad", "cold": "cold", "cool": "cool", "crazy": "crazy", "cute": "cute", "dark": "dark", "different": "different", "easy": "easy",
       "fine": "fine", "full": "full", "good": "good", "happy": "happy", "hard": "hard", "hot": "hot", "late": "late", "new": "new", "old": "old",
       "short": "short", "sick": "sick", "small": "small", "tall": "tall", "thin": "thin", "tired": "tired", "friendly": "friendly"}
FOOD_ADJ = {"delicious": "delicious", "good": "good", "bad": "bad", "hot": "hot", "cold": "cold"}
# verb: (base, past, ing, takes_object)
VERBS = {"buy": ("buy", "bought", "buying"), "cook": ("cook", "cooked", "cooking"), "bake": ("bake", "baked", "baking"), "bring": ("bring", "brought", "bringing"),
         "read": ("read", "read", "reading"), "write": ("write", "wrote", "writing"), "draw": ("draw", "drew", "drawing"), "paint": ("paint", "painted", "painting"),
         "make": ("make", "made", "making"), "need": ("need", "needed", "needing"), "want": ("want", "wanted", "wanting"), "like": ("like", "liked", "liking"),
         "lose": ("lose", "lost", "losing"), "forget": ("forget", "forgot", "forgetting"), "take": ("take", "took", "taking"), "give": ("give", "gave", "giving"),
         "check": ("check", "checked", "checking"), "copy": ("copy", "copied", "copying"), "change": ("change", "changed", "changing"), "order": ("order", "ordered", "ordering")}
INTRANS = {"dance": ("dance", "danced", "dancing"), "run": ("run", "ran", "running"), "walk": ("walk", "walked", "walking"), "work": ("work", "worked", "working"),
           "study": ("study", "studied", "studying"), "practice": ("practice", "practised", "practising"), "travel": ("travel", "travelled", "travelling"),
           "cry": ("cry", "cried", "crying"), "laugh": ("laugh", "laughed", "laughing"), "cook": ("cook", "cooked", "cooking"), "wait": ("wait", "waited", "waiting"),
           "play": ("play", "played", "playing"), "swim": None}
INTRANS = {k: v for k, v in INTRANS.items() if v}
PAST = {"yesterday": "yesterday", "past": "before", "sunday": "on Sunday", "thursday": "on Thursday", "christmas": "at Christmas", "thanksgiving": "at Thanksgiving", "halloween": "on Halloween"}
FUT = {"soon": "soon", "later": "later", "future": "in the future", "sunday": "on Sunday", "thursday": "on Thursday", "christmas": "at Christmas"}
NOW = {"now": "now"}
# which objects each verb makes sense with
FOOD_VERBS = {"buy", "cook", "bring", "make", "need", "want", "like", "order", "take", "give"}
BAKE_OK = {"apple", "banana", "cheese", "corn", "egg", "fish", "meat", "pizza", "sandwich", "tomato"}
THING_VERBS = {"buy": None, "bring": None, "need": None, "want": None, "like": None, "lose": None, "forget": None, "take": None, "give": None, "check": None,
               "read": {"book", "letter", "list", "paper"}, "write": {"letter", "list", "paper"}, "copy": {"letter", "list", "paper", "book"},
               "change": {"clothes", "shirt", "pants", "dress", "jacket", "hat", "glasses"}, "paint": {"chair", "table", "window", "bed", "paper"},
               "make": {"list", "letter", "dress", "shirt", "table", "chair"}}


def obj(v, n):
    if v in ("change", "lose", "forget", "check", "take") and n in THINGS: return "my " + n
    return art(n)


def verb_object(r):
    if r.random() < .45:
        v = r.choice(sorted(FOOD_VERBS | {"bake"}))
        n = r.choice(sorted(BAKE_OK)) if v == "bake" else r.choice(FOOD)
        if v == "cook" and n in DRINK: n = r.choice(sorted(BAKE_OK))
        return v, n
    v = r.choice(sorted(THING_VERBS)); ok = THING_VERBS[v]
    return v, r.choice(sorted(ok) if ok else THINGS)


EMOS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"]


def art(n):
    if n in DRINK or n in {"cereal", "cheese", "meat", "fish", "corn", "salt", "money", "candy", "paper", "clothes", "pants", "glasses"}: return "some " + n
    return ("an " if n[0] in "aeiou" else "a ") + n


def the(n): return n if n in {"money", "clothes", "pants", "glasses"} else "the " + n


def poss(p): return ("my " + p) if p in RELATIVE else ("the " + p)


def tone(s, emo, r, q=False):
    """Emotion is carried mainly by the voice (Chatterbox settings). Text only changes punctuation, plus 'please'
    for urgent requests under Fear/Angry; the meaning never changes."""
    excited = emo in ("Happy", "Surprise", "Angry") or (emo == "Fear" and any(w in s.lower() for w in ("help", "doctor", "police", "medicine", "accident", "hurry")))
    end = ("?!" if emo in ("Surprise", "Angry") else "?") if q else ("!" if excited else ".")
    s = s.strip(); s = s[0].upper() + s[1:]
    return s + end


def sample(r):
    k = r.randrange(14)
    if k == 0:   # WANT/NEED/LIKE + object
        v = r.choice(["want", "need", "like"]); n = r.choice(FOOD + THINGS)
        g = [n.upper(), v.upper()] if r.random() < .4 else [v.upper(), n.upper()]
        return g, f"I {v} {art(n) if v != 'like' else (n if n in DRINK else n + ('s' if n not in {'cheese','meat','fish','corn','salt','cereal','candy','money','clothes','pants','glasses'} else ''))}", False
    if k == 1:   # NO + verb + object
        v = r.choice(["want", "need", "like", "know"]); n = r.choice(FOOD + THINGS)
        if v == "know": return ["NO", "KNOW"], "I don't know", False
        return [n.upper(), "NO", v.upper()] if r.random() < .5 else ["NO", v.upper(), n.upper()], f"I don't {v} {'any ' + n if n in DRINK else the(n)}", False
    if k == 2:   # past time + verb + object
        t = r.choice(list(PAST)); v, n = verb_object(r)
        return [t.upper(), v.upper(), n.upper()], f"I {VERBS[v][1]} {obj(v, n)} {PAST[t]}", False
    if k == 3:   # future
        t = r.choice(list(FUT)); v, n = verb_object(r)
        while v in ("want", "like"): v, n = verb_object(r)
        return [t.upper(), v.upper(), n.upper()], f"I will {v} {obj(v, n)} {FUT[t]}", False
    if k == 4:   # go to place (time)
        p = r.choice(list(PLACES)); t = r.choice(["", "now", "later", "soon", "yesterday"])
        dest = PLACES[p] if p == "home" else "to " + PLACES[p]
        if t == "yesterday": return ["YESTERDAY", p.upper(), "GO"], f"I went {dest} yesterday", False
        g = ([t.upper()] if t else []) + [p.upper(), "GO"]
        return g, f"I'm going {dest}" + (f" {t}" if t else ""), False
    if k == 5:   # person + adjective
        p = r.choice(PEOPLE); a = r.choice(list(ADJ))
        return [p.upper(), a.upper()], f"{poss(p).capitalize()} is {ADJ[a]}" if p != "family" else f"My family is {ADJ[a]}", False
    if k == 6:   # food + adjective
        n = r.choice(FOOD); a = r.choice(list(FOOD_ADJ))
        return [n.upper(), a.upper()], f"The {n} is {FOOD_ADJ[a]}", False
    if k == 7:   # WH questions (WH last)
        w = r.choice(["what", "where", "who", "why", "how"])
        opts = {"what": [(["YOUR", "NAME", "WHAT"], "What is your name"), (["YOU", "WANT", "WHAT"], "What do you want"), (["YOU", "LIKE", "WHAT"], "What do you like"),
                         (["TIME", "WHAT"], "What time is it"), (["YOU", "STUDY", "WHAT"], "What are you studying"), (["YOUR", "WORK", "WHAT"], "What is your job")],
                "where": [([p.upper(), "WHERE"], f"Where is {PLACES[p] if p != 'room' else 'the room'}") for p in ["bathroom", "restaurant", "office", "school", "class"]] +
                         [(["YOU", "LIVE", "WHERE"], "Where do you live"), (["YOU", "GO", "WHERE"], "Where are you going"), ([r.choice(THINGS).upper(), "WHERE"], None)],
                "who": [(["YOUR", "TEACHER", "WHO"], "Who is your teacher"), (["DOCTOR", "WHO"], "Who is the doctor"), (["WHO", "HELP"], "Who can help")],
                "why": [(["YOU", "LATE", "WHY"], "Why are you late"), (["YOU", "CRY", "WHY"], "Why are you crying"), (["YOU", "LEAVE", "WHY"], "Why are you leaving")],
                "how": [(["YOU", "FEEL", "HOW"], "How do you feel"), (["YOU", "GO", "SCHOOL", "HOW"], "How do you get to school"), (["SIGN", "HOW"], "How do you sign that")]}
        g, s = r.choice(opts[w])
        if s is None: s = f"Where is the {g[0].lower()}"
        return g, s, True
    if k == 8:   # medical / help
        return r.choice([(["HEADACHE"], "I have a headache", False), (["NEED", "DOCTOR"], "I need a doctor", False), (["NEED", "MEDICINE"], "I need medicine", False),
                         (["SICK", "COUGH"], "I'm sick and I have a cough", False), (["PLEASE", "HELP"], "Please help me", False), (["HELP", "PLEASE", "HURRY"], "Please help me, hurry", False),
                         (["CALL", "DOCTOR"], "Please call a doctor", False), (["CALL", "POLICE"], "Please call the police", False), (["ACCIDENT", "HELP"], "There's been an accident, help", False),
                         (["FEEL", "SICK"], "I feel sick", False), (["FEEL", "FINE"], "I feel fine", False), (["TIRED"], "I'm tired", False)])
    if k == 9:   # FINISH completive
        v = r.choice(list(INTRANS) + ["eat", "drink"])
        if v in ("eat", "drink"): return [v.upper(), "FINISH"], f"I already {'ate' if v == 'eat' else 'drank'}", False
        return [v.upper(), "FINISH"], f"I've finished {INTRANS[v][2]}", False
    if k == 10:  # yes/no + simple
        return r.choice([(["YES"], "Yes", False), (["NO"], "No", False), (["THANK", "YOU"], None, False), (["YES", "PLEASE"], "Yes, please", False), (["NO", "THANKS"], None, False),
                         (["AGAIN", "PLEASE"], "Could you say that again, please", True), (["WAIT", "PLEASE"], "Please wait", False), (["SORRY", "LATE"], None, False)])
    if k == 11:  # person + verb (+place)
        p = r.choice([x for x in PEOPLE if x in RELATIVE and x != "family"]); v = r.choice(list(INTRANS))
        t = r.choice(["", "now", "yesterday"])
        if t == "yesterday": return ["YESTERDAY", p.upper(), v.upper()], f"My {p} {INTRANS[v][1]} yesterday", False
        return ([t.upper()] if t else []) + [p.upper(), v.upper()], f"My {p} is {INTRANS[v][2]}" + (" now" if t else ""), False
    if k == 12:  # possession / have
        p = r.choice(["brother", "sister", "son", "daughter", "dog", "cat", "bird", "rabbit", "car"])
        p = p if p in {"brother", "son", "daughter", "dog", "cat", "bird", "rabbit"} else "dog"
        n = r.choice(["", "two", "three"])
        if n: return ["HAVE", n.upper(), p.upper()], f"I have {n} {p}s" if p != "son" else f"I have {n} sons", False
        return ["HAVE", p.upper()], f"I have a {p}", False
    v = r.choice(["meet", "visit", "call", "help", "teach", "tell"]); p = r.choice([x for x in PEOPLE if x in RELATIVE])
    t = r.choice(["", "later", "yesterday", "soon"])
    vb = {"meet": ("meet", "met"), "visit": ("visit", "visited"), "call": ("call", "called"), "help": ("help", "helped"), "teach": ("teach", "taught"), "tell": ("tell", "told")}[v]
    if t == "yesterday": return ["YESTERDAY", v.upper(), p.upper()], f"I {vb[1]} my {p} yesterday", False
    return ([t.upper()] if t else []) + [v.upper(), p.upper()], f"I will {vb[0]} my {p}" + (f" {t}" if t else ""), False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--vocab", default="")
    a = ap.parse_args(); r = random.Random(0)
    vocab = None
    if a.vocab: vocab = {x.strip().upper() for x in open(a.vocab) if x.strip()}
    pairs, seen = [], set()
    tries = 0
    while len(pairs) < a.n and tries < a.n * 50:
        tries += 1
        g, s, q = sample(r)
        if s is None: continue
        if vocab and not all(w in vocab for w in g): continue
        emo = r.choice(EMOS)
        pairs.append(dict(gloss=" ".join(g), emotion=emo, text=tone(s, emo, r, q)))
    combos = sorted({p["gloss"] for p in pairs}); r.shuffle(combos)
    test_combos = set(combos[: max(1, len(combos) // 10)])   # held-out word combinations, never seen in training
    tr = [p for p in pairs if p["gloss"] not in test_combos]; te = [p for p in pairs if p["gloss"] in test_combos]
    os.makedirs("data", exist_ok=True)
    for name, rows in (("train", tr), ("test", te)):
        with open(f"data/gloss_en_{name}.jsonl", "w") as f:
            for p in rows: f.write(json.dumps(p) + "\n")
    with open("data/gloss_en_review.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gloss", "emotion", "generated_english", "ok? (y/n)", "better English"])
        for p in r.sample(tr, min(300, len(tr))): w.writerow([p["gloss"], p["emotion"], p["text"], "", ""])
    print(f"{len(tr)} train, {len(te)} test pairs; {len(combos)} distinct gloss sequences ({len(test_combos)} held out)")


if __name__ == "__main__":
    main()
