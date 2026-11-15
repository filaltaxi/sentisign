"""Rule baseline gloss->English (what a fine-tuned Flan-T5 must beat) + TTS emotion map."""
import re
TTS = {"Happy": (0.65, 0.25), "Angry": (0.82, 0.68), "Sad": (0.22, 0.88), "Fear": (0.68, 0.82),
       "Disgust": (0.55, 0.52), "Surprise": (0.78, 0.22), "Neutral": (0.38, 0.32)}  # (exaggeration, cfg)
PRON = {"ME": "I", "I": "I", "YOU": "you", "HE": "he", "SHE": "she", "WE": "we", "THEY": "they"}
VERB = {"WANT", "LIKE", "NEED", "HAVE", "GO", "EAT", "DRINK", "HELP", "LOVE", "KNOW", "SEE", "WORK", "LEARN"}
Q = {"WHAT", "WHERE", "WHO", "WHY", "HOW", "WHEN"}


def baseline(glosses, question=False):
    g = [x.upper() for x in glosses]
    if not g: return ""
    out = []
    for i, w in enumerate(g):
        if w in PRON: out.append(PRON[w])
        elif w == "NOT": out.append("don't" if any(x in VERB for x in g) else "not")
        else: out.append(w.lower())
    s = " ".join(out); s = s[0].upper() + s[1:]
    if question or g[0] in Q or g[-1] in Q: return s + "?"
    return s + "."


def tts_params(emotion): return TTS.get(emotion, TTS["Neutral"])
