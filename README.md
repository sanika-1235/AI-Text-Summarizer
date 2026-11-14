# Mini Project 1: AI Text Summarizer

An end-to-end, controllable text summarization pipeline built with Python and Hugging Face's instruction-tuned **`google/flan-t5-base`** model, engineered to run entirely on CPU. The project demonstrates dual-steering length control (prompting plus decoding constraints), resilient input validation, sentence-aware Map-Reduce chunking for long documents, and an accountable factual evaluation suite comprising automated hallucination detection, ROUGE metric scoring, and a qualitative manual rubric.

---

## Key Features

- **Pretrained Instruction-Tuned Transformer:** Uses `google/flan-t5-base` (~250M parameters) running locally on CPU without requiring GPU hardware.
- **Dual-Steering Length Control:** Seamlessly switch between length presets (`'short'`, `'medium'`, `'long'`) or supply an exact custom word count (e.g. `45`), enforced both at the prompt level and via calibrated generation bounds (`min_new_tokens`, `max_new_tokens`, `num_beams`, `no_repeat_ngram_size`).
- **Production-Grade Input Validation:** Gracefully intercepts `None`, whitespace, non-string types, very short texts (< 20 words), and texts exceeding a hard 25,000-character ceiling without crashing.
- **Map-Reduce for Long Texts:** Automatically detects inputs exceeding the model's token limits, splits them at sentence boundaries into sub-chunks, summarizes each chunk, and synthesizes a cohesive final summary.
- **Two-Tier Factual Accuracy Evaluation:**
  - *Automated Checks:* Verifies whether numbers, dates, currency amounts, percentages, and named entities in the summary are grounded in the source text (flagging hallucinations), and computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores against human reference summaries.
  - *Manual Rubric Audit:* Qualitative evaluation table in pandas detailing factual faithfulness, hallucinated facts, missed information, accuracy scores (1–5), and explanatory notes across 5 diverse domain texts (News, Science, Business, History, Tech).

---

## Project Structure

```text
WEEK 1/
├── summarizer.ipynb     # Complete, self-contained Jupyter notebook (all 12 required sections)
└── README.md            # Project documentation and execution instructions
```

---

## Setup & Execution

### 1. Requirements
- Python 3.10+ (tested on Python 3.14)
- CPU with 4GB+ RAM

### 2. Installation
Install the necessary dependencies in your Python environment:

```bash
pip install transformers torch sentencepiece rouge-score pandas
```

### 3. Running the Notebook
Launch Jupyter Notebook or open the file in VS Code / JupyterLab:

```bash
jupyter notebook summarizer.ipynb
```

Run all cells sequentially from top to bottom (**Kernel -> Restart and Run All**). Every cell has been pre-validated to run without errors.

---

## Evaluation Summary & Findings

Evaluating five diverse passages (News, Science, Business, History, Technology) yielded an **average factual accuracy score of 3.6 / 5.0**:

- **Factual Fidelity (Score ≥ 4):** 60% of test cases (CRISPR science explainer, SwiftLogistics robotics business update, and RAG technical architecture) demonstrated high factual grounding with zero hallucinations.
- **Pretraining Prior Contamination (Example 1):** When summarizing a municipal power grid announcement in Greenville, the model hallucinated the state *"South Carolina"*, demonstrating how strong pretraining associations can leak into abstractive summaries under short length constraints.
- **Entity Conflation & Timeline Confusion (Example 4):** Summarizing dense historical passages revealed susceptibility to semantic drift: the model conflated Theodore Roosevelt with *"Franklin D. Roosevelt"* and generated contradictory timeline phrases (*"completed in 1904 and completed in 1914"*).
- **Automated Hallucination Detection:** Regex-based entity and number grounding caught 100% of the hallucinated dates and entities, proving to be an essential safety layer for generative summarization pipelines.

---

## Limitations & Future Work

1. **NLI Faithfulness Filters:** Incorporate Natural Language Inference (e.g. `roberta-large-mnli`) to reject or regenerate summaries with premise-hypothesis contradiction scores.
2. **Constrained Decoding:** Enforce grammar- and entity-constrained decoding (e.g. via `outlines`) to ensure all generated proper nouns originate strictly from the input context.
3. **Interactive UI:** Wrap the pipeline into a **Streamlit** or **Gradio** web app featuring interactive length sliders and real-time hallucination highlighting.

