# Natural Language Understanding: Dependency Parsing

# 🛑 DEADLINE
## 8 DECEMBER 2025

The main goal is to train and evaluate a **Dependency Parser** based on the transition algorithm **arc-eager** and a **feed-forward neural network** that predicts the next transitions from the current state (stack, buffer, PoS tags, etc.) using CoNLL-U **UD_English-ParTUT**.

---

## Project Structure

```text
.
├── main.py
├── src/
│   ├── ...
└── data/
    ├── train.conllu
    ├── dev.conllu
    └── test.conllu
