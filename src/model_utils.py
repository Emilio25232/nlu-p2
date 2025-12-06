"""
Utility functions for loading and saving the ParserMLP model.
"""

import pickle
from src.model import ParserMLP


def load_model(weights_path="models/parser_mlp.weights.h5",
               config_path="models/model_config.pkl",
               dictionaries_path="models/dictionaries.pkl"):
    """
    Load a trained ParserMLP model from saved weights and configuration.
    
    Parameters:
        weights_path (str): Path to the saved model weights.
        config_path (str): Path to the saved model configuration.
        dictionaries_path (str): Path to the saved dictionaries.
    
    Returns:
        model (ParserMLP): The loaded model.
        dictionaries (dict): Dictionary containing all vocabularies.
    """
    # Load model configuration
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    # Load dictionaries
    with open(dictionaries_path, "rb") as f:
        dictionaries = pickle.load(f)
    
    # Create model with loaded configuration
    model = ParserMLP(
        word_emb_dim=config['word_emb_dim'],
        pos_emb_dim=config['pos_emb_dim'],
        hidden_dim=config['hidden_dim'],
        vocab_size_form=config['vocab_size_form'],
        vocab_size_upos=config['vocab_size_upos'],
        n_actions=config['n_actions'],
        n_deprels=config['n_deprels'],
        n_word_feats=config['n_word_feats'],
        n_pos_feats=config['n_pos_feats']
    )
    
    # Load weights
    model.model.load_weights(weights_path)
    
    print(f"✓ Model loaded from {weights_path}")
    print(f"✓ Dictionaries loaded from {dictionaries_path}")
    
    return model, dictionaries


def save_model(model, dictionaries, 
               weights_path="models/parser_mlp.weights.h5",
               config_path="models/model_config.pkl",
               dictionaries_path="models/dictionaries.pkl"):
    """
    Save a ParserMLP model's weights, configuration, and dictionaries.
    
    Parameters:
        model (ParserMLP): The model to save.
        dictionaries (dict): Dictionary containing all vocabularies.
        weights_path (str): Path to save the model weights.
        config_path (str): Path to save the model configuration.
        dictionaries_path (str): Path to save the dictionaries.
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    # Save weights
    model.model.save_weights(weights_path)
    
    # Save configuration
    model_config = {
        'word_emb_dim': model.word_emb_dim,
        'pos_emb_dim': model.pos_emb_dim,
        'hidden_dim': model.hidden_dim,
        'vocab_size_form': model.vocab_size_form,
        'vocab_size_upos': model.vocab_size_upos,
        'n_actions': model.n_actions,
        'n_deprels': model.n_deprels,
        'n_word_feats': model.n_word_feats,
        'n_pos_feats': model.n_pos_feats
    }
    
    with open(config_path, "wb") as f:
        pickle.dump(model_config, f)
    
    # Save dictionaries
    with open(dictionaries_path, "wb") as f:
        pickle.dump(dictionaries, f)
    
    print(f"✓ Model weights saved to {weights_path}")
    print(f"✓ Model config saved to {config_path}")
    print(f"✓ Dictionaries saved to {dictionaries_path}")
