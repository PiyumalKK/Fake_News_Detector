# Fake News Detection — Hybrid (Local ML + Azure Foundry)

**Course:** Advanced Artificial Intelligence (EE7260)
**University of Ruhuna — Department of Electrical and Information Engineering**

Detect whether a news article is **REAL** or **FAKE** using three different AI
approaches and compare them. Local Python does the heavy ML; Azure AI Foundry
provides the Large Language Model used for prompt-engineering.

---

## AI techniques used

| # | Technique | How we implement it | Where it runs |
|---|-----------|---------------------|---------------|
| 1 | **NLP** | Text cleaning, tokenisation, lemmatisation, stop-word removal, TF-IDF features, word embeddings | Local Python |
| 2 | **Transformer fine-tuning / transfer learning** | Fine-tune **DistilBERT** on the dataset (Hugging Face) | Local Python |
| 3 | **Prompt Engineering** | **GPT-4o** zero-shot, few-shot, and chain-of-thought classification | **Azure AI Foundry** |

DistilBERT is fine-tuned locally because Foundry fine-tunes GPT-family models
rather than BERT. Foundry hosts the LLM used for the prompt-engineering approach.

### Model comparison

```
Article text ──┬─► Baseline:  TF-IDF + Logistic Regression   (classic ML)
               ├─► Model 2:   Fine-tuned DistilBERT          (transformer)
               └─► Model 3:   GPT-4o few-shot + CoT          (prompt engineering, Foundry)
```

All three are evaluated with **accuracy, precision, recall, F1, confusion matrix,
and ROC-AUC**, then compared in one table.

---

## Project structure

```
FakeNewsDetection/
├── README.md                 ← you are here
├── requirements.txt          ← Python dependencies
├── .env.example              ← copy to .env and fill Foundry keys
├── .gitignore
├── data/                     ← datasets (not committed)
├── notebooks/
│   ├── 01_preprocessing_baseline.ipynb   ← Week 1
│   ├── 02_distilbert_finetune.ipynb      ← Week 2
│   ├── 03_prompt_engineering.ipynb       ← Week 2 (Foundry)
│   └── 04_evaluation_comparison.ipynb    ← Week 3
├── app/
│   └── gradio_app.py         ← Week 3 demo
├── docs/
│   └── foundry-setup.md      ← portal steps for the Foundry part
└── results/                  ← saved metrics, plots, models
```


## Setup (one time)

```powershell
# from this folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then for the Foundry part (Week 2): follow [docs/foundry-setup.md](docs/foundry-setup.md),
copy `.env.example` to `.env`, and paste your endpoint + key.

---
