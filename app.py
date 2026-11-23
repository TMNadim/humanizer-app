import hashlib
import re

import requests
import streamlit as st

URL = st.secrets["MODAL_URL"].rstrip("/")
KEY = st.secrets["API_KEY"]

INSTR = (
    "Rewrite the text below so it reads like a person wrote it, not a language model.\n\n"
    "Reorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\n"
    "throat-clearing, and any sentence that only announces what comes next.\n"
    "Prefer the concrete word over the abstract one. It is fine to sound uneven.\n\n"
    "Every fact, number, unit, date, name and quotation must survive unchanged."
)
SEP = "\n\n### Rewritten:\n\n"


def build_prompt(draft: str) -> str:
    return INSTR + "\n\n" + draft.strip() + SEP


# Safety check: stops the app if the prompt ever differs from what the model expects.
assert hashlib.sha256(build_prompt("X").encode("utf-8")).hexdigest()[:16] == "cc51d66b4c593fbe"


def chunk(text, max_words=350):
    paras = re.split(r"\n\s*\n", text.strip())
    chunks, cur = [], ""
    for p in paras:
        if cur and len((cur + " " + p).split()) > max_words:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def has_chinese(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s))


def call_model(draft):
    r = requests.post(
        f"{URL}/completion",
        headers={"Authorization": f"Bearer {KEY}"},
        json={
            "prompt": build_prompt(draft),
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0,
            "repeat_penalty": 1.0,
            "n_predict": 2048,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["content"].strip()


def humanize(draft):
    draft_is_english = has_chinese(draft) < 5
    out = ""
    for _ in range(3):  # retry if the model wrongly answers in Chinese
        out = call_model(draft)
        if not (draft_is_english and has_chinese(out) > 3):
            break
    return out


def numbers(s):
    return set(re.findall(r"\d[\d,\.]*", s))


st.set_page_config(page_title="Text Humanizer")
st.title("Text Humanizer")
st.caption("Rewrites AI-sounding drafts so they read more naturally. English and Chinese.")

text = st.text_area("Paste your draft", height=300)
st.caption(f"{len(text.split())} words")

if st.button("Humanize", type="primary"):
    if len(text.split()) < 50:
        st.info("Please paste at least 50 words. The model wasn't trained on very short text.")
    else:
        parts = chunk(text)
        out = []
        bar = st.progress(0, text="Waking the model. The first run can take a few minutes...")
        for i, part in enumerate(parts):
            out.append(humanize(part))
            bar.progress((i + 1) / len(parts), text=f"Chunk {i + 1} of {len(parts)}")
        result = "\n\n".join(out)
        st.text_area("Result", result, height=300)
        st.download_button("Download .txt", result, "humanized.txt")

        missing = sorted(numbers(text) - numbers(result))
        if missing:
            st.warning(f"Please double-check these numbers, which I couldn't find in the result: {', '.join(missing)}")
        st.caption("Always proofread names, dates, numbers and the meaning of each claim.")