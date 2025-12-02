from collections import Counter
from typing import Dict, List, Tuple
from src.conllu.conllu_token import Token
from src.algorithm import Sample, ArcEager

PAD = "<PAD>"
UNK = "<UNK>"

def build_form_upos_deprel_vocabs(
    trees: List[List[Token]],
    min_freq: int = 1
) -> Tuple[Dict[str, int], Dict[int, str],
           Dict[str, int], Dict[int, str],
           Dict[str, int], Dict[int, str]]:
    """
    Build vocabularies for FORM, UPOS and DEPREL from the training trees.

    Args:
        trees: list of sentences, each sentence is a list[Token]
        min_freq: minimum frequency to keep a word (FORM) in the vocab

    Returns:
        form2id, id2form, upos2id, id2upos, deprel2id, id2deprel
    """
    form_counter = Counter()
    upos_set = set()
    deprel_set = set()

    for sent in trees:
        for tok in sent:
            # FORM
            if tok.form is not None:
                form_counter[tok.form] += 1
            # UPOS
            if tok.upos is not None:
                upos_set.add(tok.upos)
            # DEPREL (dependency label)
            if tok.dep is not None and tok.id != 0:  # skip ROOT if you want
                deprel_set.add(tok.dep)

    # FORM vocab with PAD and UNK
    form2id: Dict[str, int] = {}
    id2form: Dict[int, str] = {}

    # Reserve 0 and 1 for special tokens
    form2id[PAD] = 0
    id2form[0] = PAD
    form2id[UNK] = 1
    id2form[1] = UNK

    idx = 2
    for form, freq in form_counter.items():
        if freq >= min_freq and form not in form2id:
            form2id[form] = idx
            id2form[idx] = form
            idx += 1

    # UPOS vocab
    upos2id: Dict[str, int] = {}
    id2upos: Dict[int, str] = {}

    upos2id[PAD] = 0
    id2upos[0] = PAD
    idx = 1
    for up in sorted(upos_set):
        if up not in upos2id:
            upos2id[up] = idx
            id2upos[idx] = up
            idx += 1

    # DEPREL vocab
    deprel2id: Dict[str, int] = {}
    id2deprel: Dict[int, str] = {}

    deprel2id[PAD] = 0
    id2deprel[0] = PAD
    idx = 1
    for d in sorted(deprel_set):
        if d not in deprel2id:
            deprel2id[d] = idx
            id2deprel[idx] = d
            idx += 1

    return form2id, id2form, upos2id, id2upos, deprel2id, id2deprel


def build_transition_vocab(
    samples: List[Sample]
) -> Tuple[Dict[Tuple[str, str], int], Dict[int, Tuple[str, str]]]:
    """
    Build a vocabulary over transitions (action, dependency_label) pairs.

    Example keys:
        ("SHIFT", None)
        ("REDUCE", None)
        ("LEFT-ARC", "nsubj")
        ("RIGHT-ARC", "obj")
    """
    trans_set = set()
    for sample in samples:
        t = sample.transition
        key = (t.action, t.dependency)
        trans_set.add(key)

    trans2id: Dict[Tuple[str, str], int] = {}
    id2trans: Dict[int, Tuple[str, str]] = {}

    for idx, key in enumerate(sorted(trans_set)):
        trans2id[key] = idx
        id2trans[idx] = key

    return trans2id, id2trans


def build_action_only_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Simpler alternative: only classify the action type (SHIFT, LA, RA, REDUCE),
    and predict dependency labels with a separate head.

    Returns:
        action2id, id2action
    """
    actions = [ArcEager.SHIFT, ArcEager.LA, ArcEager.RA, ArcEager.REDUCE]

    action2id: Dict[str, int] = {}
    id2action: Dict[int, str] = {}

    for idx, a in enumerate(actions):
        action2id[a] = idx
        id2action[idx] = a

    return action2id, id2action
