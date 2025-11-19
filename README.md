# Natural Language Understanding: Dependency Parsing

The main goal is train and evalutate a **Dependency Parsing** based of the transition algorith **arc-eager** and a **feed-forward neural networkk** that predicts the next transitions from the actual state (stack, buffer, PoS tags, etc.) from CoNLL-U **UD_English-ParTUT**.

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
```

## Dependencies

Libraries used: 
- `tensorflow`