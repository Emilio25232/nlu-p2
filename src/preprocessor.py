# src/data_utils.py
from typing import List, Tuple
import numpy as np

from src.algorithm import Sample, ArcEager
from src.vocab import PAD, UNK


def samples_to_arrays(
    samples: List[Sample],
    form2id,
    upos2id,
    deprel2id,
    action2id,
    nbuffer_feats: int = 2,
    nstack_feats: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a list of Sample is NumPy matrix for Keras.
    """
    n_word_feats = nstack_feats + nbuffer_feats

    X_words = []
    X_pos = []
    y_action = []
    y_deprel = []

    pad_id_form = form2id[PAD]
    unk_id_form = form2id[UNK]
    pad_id_pos = upos2id[PAD]

    for sample in samples:
        feats = sample.state_to_feats(
            nbuffer_feats=nbuffer_feats,
            nstack_feats=nstack_feats,
        )
        words = feats[:n_word_feats]
        pos_tags = feats[n_word_feats:]

        word_ids = [
            form2id.get(w, unk_id_form)
            for w in words
        ]

        pos_ids = [
            upos2id.get(p, pad_id_pos)
            for p in pos_tags
        ]

        t = sample.transition
        action_id = action2id[t.action]

        if t.action in (ArcEager.LA, ArcEager.RA):
            deprel_id = deprel2id[t.dependency]
        else:
            deprel_id = -1

        X_words.append(word_ids)
        X_pos.append(pos_ids)
        y_action.append(action_id)
        y_deprel.append(deprel_id)

    X_words = np.asarray(X_words, dtype="int32")
    X_pos = np.asarray(X_pos, dtype="int32")
    y_action = np.asarray(y_action, dtype="int32")
    y_deprel = np.asarray(y_deprel, dtype="int32")

    return X_words, X_pos, y_action, y_deprel
