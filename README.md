# Arabic POS Tagging using asafaya/bert-base-arabic

This project builds a Part-of-Speech (POS) tagging model for Arabic using the Hugging Face `transformers` library and asafaya/bert-base-arabic , trained on a CoNLL-U formatted dataset.

---

## 📌 Steps Overview

1. **Data Loading & Preprocessing**
   - Loaded Arabic POS dataset from `.conllu` format using `pyconll`.
   - Extracted tokens and their corresponding UPOS tags.

2. **Tokenization & Label Alignment**
   - Used Hugging Face tokenizer to split text.
   - Aligned word-level UPOS tags with tokenized subwords.
   - Created `label2id` and `id2label` mappings.

3. **Dataset Preparation**
   - Converted data to Hugging Face `Dataset`.
   - Applied formatting for token classification (including `input_ids`, `attention_mask`, and `labels`).

4. **Dataset Splitting**
   - Train: 70%
   - Validation: 15%
   - Test: 15%

5. **Model Training**
   - Used `BertForTokenClassification`.
   - Evaluation metrics via `seqeval`.
   - Tracked performance via `wandb`.

6. **Evaluation**
   - Printed formatted predictions with token, tag, and score.
   - Used `classification_report` for final evaluation.

---

## 🔧 Dependencies

```bash
pip install pyconll datasets transformers seqeval tabulate accelerate wandb
