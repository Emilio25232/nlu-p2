
from src.conllu.conllu_reader import ConlluReader
from src.algorithm import ArcEager
from src.vocab import (
    build_form_upos_deprel_vocabs,
    build_action_only_vocab,
)
from src.preprocessor import samples_to_arrays


def read_file(reader, path, inference):
    trees = reader.read_conllu_file(path, inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
    for token in trees[0]:
        print (token)
    print ()
    return trees


reader = ConlluReader()
train_trees = read_file(reader,path="data/en_partut-ud-train_clean.conllu", inference=False)
dev_trees = read_file(reader,path="data/en_partut-ud-dev_clean.conllu", inference=False)
test_trees = read_file(reader,path="data/en_partut-ud-test_clean.conllu", inference=True)

train_trees = reader.remove_non_projective_trees(train_trees)
dev_trees = reader.remove_non_projective_trees(dev_trees)

arc_eager = ArcEager()

form2id, id2form, upos2id, id2upos, deprel2id, id2deprel = \
    build_form_upos_deprel_vocabs(train_trees, min_freq=1)

train_samples = []
for sent in train_trees:
    train_samples.extend(arc_eager.oracle(sent))

dev_samples = []
for sent in dev_trees:
    dev_samples.extend(arc_eager.oracle(sent))

action2id, id2action = build_action_only_vocab()

# Samples → arrays (Issue 7)
X_train_words, X_train_pos, y_train_action, y_train_deprel = samples_to_arrays(
    train_samples, form2id, upos2id, deprel2id, action2id
)

X_dev_words, X_dev_pos, y_dev_action, y_dev_deprel = samples_to_arrays(
    dev_samples, form2id, upos2id, deprel2id, action2id
)

# TODO: Complete the ArcEager algorithm class.
# DONE 1. Implement the 'oracle' function and auxiliary functions to determine the correct parser actions.
#    Note: The SHIFT action is already implemented as an example.
#    Additional Note: The 'create_initial_state()', 'final_state()', and 'gold_arcs()' functions are already implemented.
# DONE 2. Use the 'oracle' function in ArcEager to generate all training samples, creating a dataset for training the neural model.
# DONE 3. Utilize the same 'oracle' function to generate development samples for model tuning and evaluation.

# TODO: Implement the 'state_to_feats' function in the Sample class.
# DONE This function should convert the current parser state into a list of features for use by the neural model classifier.

# TODO: Define and implement the neural model in the 'model.py' module.
# 1. Train the model on the generated training dataset.
# 2. Evaluate the model's performance using the development dataset.
# 3. Conduct inference on the test set with the trained model.
# 4. Save the parsing results of the test set in CoNLLU format for further analysis.

# TODO: Utilize the 'postprocessor' module (already implemented).
# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"
# 3. Process the file: trees = postprocessor.postprocess(path)
# 4. Save the processed trees to a new output file.