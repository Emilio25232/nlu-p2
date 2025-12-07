
from src.conllu.conllu_reader import ConlluReader
from src.algorithm import ArcEager
from src.model import ParserMLP
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

print("\n" + "="*60)
print("TRAINING THE MLP DEPENDENCY PARSER")
print("="*60)

# Print dataset statistics
print(f"\n Dataset Statistics:")
print(f"   Training samples: {len(train_samples):,}")
print(f"   Dev samples: {len(dev_samples):,}")
print(f"   Vocabulary sizes:")
print(f"      - Forms (words): {len(form2id):,}")
print(f"      - UPOS tags: {len(upos2id):,}")
print(f"      - Dependency relations: {len(deprel2id):,}")
print(f"      - Actions: {len(action2id)}")

# Create the model with vocabulary sizes
print(f"\n🔧 Building the MLP model...")
model = ParserMLP(
    word_emb_dim=100,
    pos_emb_dim=25,
    hidden_dim=64,
    epochs=10,
    batch_size=64,
    vocab_size_form=len(form2id),
    vocab_size_upos=len(upos2id),
    n_actions=len(action2id),
    n_deprels=len(deprel2id),
    n_word_feats=4,
    n_pos_feats=4
)

print(f"   Model architecture:")
print(f"      - Word embedding dim: 100")
print(f"      - POS embedding dim: 25")
print(f"      - Hidden layer dim: 64")
print(f"      - Output actions: {len(action2id)}")
print(f"      - Output deprels: {len(deprel2id)}")

# Print model summary
print(f"\n Model Summary:")
model.model.summary()

# Train the model
print(f"\n Training the model...")
print(f"   Epochs: {model.epochs}")
print(f"   Batch size: {model.batch_size}")
print(f"\n" + "-"*60)

history = model.train(
    X_train_words, X_train_pos, y_train_action, y_train_deprel,
    X_dev_words, X_dev_pos, y_dev_action, y_dev_deprel
)

print("-"*60)
print("\n Training completed!")

# Print training results
print(f"\n Training Results:")
print(f"   Final training action accuracy: {history.history['output_action_accuracy'][-1]:.4f}")
print(f"   Final training deprel accuracy: {history.history['output_deprel_accuracy'][-1]:.4f}")
print(f"   Final validation action accuracy: {history.history['val_output_action_accuracy'][-1]:.4f}")
print(f"   Final validation deprel accuracy: {history.history['val_output_deprel_accuracy'][-1]:.4f}")

# Evaluate the model on dev set with detailed metrics
print(f"\n Evaluating the model on dev set...")
from src.evaluator import evaluate_model_on_dev, print_evaluation_metrics

dev_metrics = evaluate_model_on_dev(
    model, X_dev_words, X_dev_pos, y_dev_action, y_dev_deprel,
    batch_size=model.batch_size
)

print_evaluation_metrics(dev_metrics, dataset_name="Development")

# Save trained weights and dictionaries
print(f"\n Saving model, weights, and vocabularies...")
from src.model_utils import save_model

dictionaries = {
    'form2id': form2id,
    'id2form': id2form,
    'upos2id': upos2id,
    'id2upos': id2upos,
    'deprel2id': deprel2id,
    'id2deprel': id2deprel,
    'action2id': action2id,
    'id2action': id2action
}

save_model(model, dictionaries)

print(f"\n" + "="*60)
print(" TRAINING COMPLETE!")
print("="*60)

# Run inference on test set
print(f"\n Running inference on test set...")
parsed_trees = model.run(
    test_trees, arc_eager, form2id, upos2id, id2action, id2deprel,
    nbuffer_feats=2, nstack_feats=2
)
print(f" Parsed {len(parsed_trees)} test sentences")

# Save results to CoNLLU format (you'll need to implement a writer function)
output_path = "data/output.conllu"
reader.write_conllu_file(parsed_trees, output_path)
print(f' Predicted trees written to "{output_path}"')

# TODO: Utilize the 'postprocessor' module (already implemented).
# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"
# 3. Process the file: trees = postprocessor.postprocess(path)
# 4. Save the processed trees to a new output file.