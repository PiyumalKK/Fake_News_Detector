"""Fake News Detection - interactive demo.

Loads whichever models are available and lets you paste an article to get a
REAL / FAKE prediction from each one:

  - TF-IDF + classic ML baseline      (always, from results/models/)
  - Fine-tuned DistilBERT             (if results/models/distilbert exists)
  - GPT-4o prompt engineering         (if a .env with Foundry keys is present)

Run:
    python app/gradio_app.py
then open the local URL it prints.
"""
import os
import re
from pathlib import Path

import gradio as gr
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "results" / "models"
LABEL_NAMES = {0: "FAKE", 1: "REAL"}

# --- text cleaning (must match the baseline notebook) -----------------------
import nltk
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()
URL_RE = re.compile(r"http\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text):
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = [LEMM.lemmatize(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(tokens)


# --- load baseline ----------------------------------------------------------
baseline_model = None
vectorizer = None
try:
    baseline_model = joblib.load(MODELS / "baseline_model.joblib")
    vectorizer = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    print("Loaded baseline model.")
except Exception as e:
    print("Baseline model not available:", e)


def predict_baseline(text):
    if baseline_model is None:
        return "baseline not trained yet"
    vec = vectorizer.transform([clean_text(text)])
    pred = int(baseline_model.predict(vec)[0])
    return LABEL_NAMES[pred]


# --- load DistilBERT (optional) --------------------------------------------
bert_model = None
bert_tokenizer = None
bert_dir = MODELS / "distilbert"
if bert_dir.exists():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
        bert_model.eval()
        print("Loaded DistilBERT model.")
    except Exception as e:
        print("DistilBERT not available:", e)


def predict_bert(text):
    if bert_model is None:
        return "DistilBERT not trained yet"
    import torch

    enc = bert_tokenizer(text[:2000], truncation=True, padding=True,
                         max_length=192, return_tensors="pt")
    with torch.no_grad():
        logits = bert_model(**enc).logits
    pred = int(logits.argmax(dim=1).item())
    return LABEL_NAMES[pred]


# --- Foundry LLM (optional) -------------------------------------------------
llm_client = None
deployment = None
try:
    from dotenv import load_dotenv
    from openai import AzureOpenAI

    load_dotenv(ROOT / ".env")
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
        llm_client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
        print("Loaded Foundry client.")
except Exception as e:
    print("Foundry client not available:", e)


def predict_llm(text):
    if llm_client is None:
        return "Foundry not configured (.env missing)"
    messages = [
        {"role": "system", "content": "You decide whether a news article is REAL or FAKE."},
        {"role": "user", "content": (
            "Classify the article. First give one short reasoning sentence, then on the "
            "final line write exactly: LABEL: REAL or LABEL: FAKE.\n\nARTICLE:\n" + text[:1500])},
    ]
    try:
        resp = llm_client.chat.completions.create(
            model=deployment, messages=messages, temperature=0, max_tokens=120)
        reply = resp.choices[0].message.content
        m = re.search(r"LABEL:\s*(REAL|FAKE)", reply.upper())
        label = m.group(1) if m else ("FAKE" if "FAKE" in reply.upper() else "REAL")
        return f"{label}  -  {reply.strip()[:160]}"
    except Exception as e:
        return f"error: {e}"


# --- UI ---------------------------------------------------------------------
def analyse(text):
    if not text or not text.strip():
        return "-", "-", "-"
    return predict_baseline(text), predict_bert(text), predict_llm(text)


EXAMPLE_REAL = ("The central bank announced on Tuesday that interest rates would remain "
                "unchanged, citing stable inflation figures released by the statistics office.")
EXAMPLE_FAKE = ("BREAKING: Scientists reveal that drinking this one common beverage makes you "
                "immortal - the government has been hiding this miracle for years!!!")

with gr.Blocks(title="Fake News Detection") as demo:
    gr.Markdown("# Fake News Detection\nPaste a news article and compare what each model predicts.")
    inp = gr.Textbox(lines=10, label="News article text")
    btn = gr.Button("Analyse", variant="primary")
    with gr.Row():
        out_base = gr.Textbox(label="Baseline (TF-IDF + ML)")
        out_bert = gr.Textbox(label="DistilBERT")
        out_llm = gr.Textbox(label="GPT-4o (Foundry)")
    gr.Examples([[EXAMPLE_REAL], [EXAMPLE_FAKE]], inputs=inp)
    btn.click(analyse, inputs=inp, outputs=[out_base, out_bert, out_llm])

if __name__ == "__main__":
    demo.launch()
