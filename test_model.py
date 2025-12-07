"""
Test script for loading and using a trained dependency parser model.

This script demonstrates how to:
1. Load a previously trained model
2. Run inference on test data
3. Evaluate performance

Usage:
    python test_model.py
"""

from src.model_utils import load_model
from src.conllu.conllu_reader import ConlluReader
from src.algorithm import ArcEager

def main():
    print("="*60)
    print("LOADING TRAINED MODEL")
    print("="*60)
    
    # Load trained model and dictionaries
    print("\n Loading model from disk...")
    model, dictionaries = load_model(
        weights_path="models/parser_mlp_weights.h5",
        config_path="models/model_config.pkl",
        dictionaries_path="models/dictionaries.pkl"
    )
    
    # Extract dictionaries
    form2id = dictionaries['form2id']
    id2form = dictionaries['id2form']
    upos2id = dictionaries['upos2id']
    id2upos = dictionaries['id2upos']
    deprel2id = dictionaries['deprel2id']
    id2deprel = dictionaries['id2deprel']
    action2id = dictionaries['action2id']
    id2action = dictionaries['id2action']
    
    print(f"\n Model loaded successfully!")
    print(f"   Vocabulary sizes:")
    print(f"      - Forms: {len(form2id):,}")
    print(f"      - UPOS tags: {len(upos2id):,}")
    print(f"      - Dependency relations: {len(deprel2id):,}")
    print(f"      - Actions: {len(action2id)}")
    
    # Load test data
    print(f"\n Loading test data...")
    reader = ConlluReader()
    test_trees = reader.read_conllu_file("data/en_partut-ud-test_clean.conllu", inference=True)
    print(f"   ✓ Loaded {len(test_trees)} test sentences")
    
    # Create ArcEager instance
    arc_eager = ArcEager()
    
    # Run inference
    print(f"\n Running inference on test set...")
    parsed_trees = model.run(
        test_trees, 
        arc_eager, 
        form2id, 
        upos2id, 
        id2action, 
        id2deprel,
        nbuffer_feats=2, 
        nstack_feats=2
    )
    
    print(f"   ✓ Successfully parsed {len(parsed_trees)} sentences")
    
    # Display a sample parsed sentence
    print(f"\n Sample parsed sentence:")
    if parsed_trees and len(parsed_trees) > 0:
        sample_sent = parsed_trees[0]
        print(f"\n   Sentence: {' '.join([token.form for token in sample_sent[1:]])}")
        print(f"\n   Token details:")
        for token in sample_sent[1:6]:  # Show first 5 tokens
            print(f"      {token.id:2d}. {token.form:15s} {token.upos:6s} -> head: {token.head:2d}")
    
    print(f"\n" + "="*60)
    print(" INFERENCE COMPLETE!")
    print("="*60)
    
    return parsed_trees


if __name__ == "__main__":
    parsed_trees = main()
