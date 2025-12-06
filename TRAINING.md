# Training the Dependency Parser

This document explains how to train and use the MLP-based dependency parser.

## Training Pipeline

The training pipeline is implemented in `main.py` and includes the following steps:

### 1. Data Preparation
- Load training, dev, and test sets from CoNLLU files
- Remove non-projective trees
- Build vocabularies for FORM, UPOS, and DEPREL
- Generate oracle samples using the Arc-Eager algorithm
- Convert samples to tensors (X_words, X_pos, y_action, y_deprel)

### 2. Model Architecture
The MLP model consists of:
- **Input layers**: Two inputs for word features and POS features
- **Embedding layers**: 
  - Word embeddings (dim: 100)
  - POS embeddings (dim: 25)
- **Hidden layers**: Two dense layers with ReLU activation (dim: 64)
- **Output layers**:
  - Action classifier (4 actions: SHIFT, LEFT-ARC, RIGHT-ARC, REDUCE)
  - Dependency relation classifier (deprel labels)

### 3. Training Configuration
- **Optimizer**: Adam
- **Loss**: Sparse categorical cross-entropy for both outputs
- **Metrics**: Accuracy for actions and deprels
- **Epochs**: 10 (configurable)
- **Batch size**: 64 (configurable)

### 4. Model Outputs
The training script automatically saves:
- `models/parser_mlp_weights.h5` - Trained model weights
- `models/model_config.pkl` - Model configuration
- `models/dictionaries.pkl` - All vocabularies (form2id, upos2id, deprel2id, action2id)

## Running the Training

Simply execute:
```bash
python main.py
```

The script will:
1. Load and preprocess the data
2. Build the model architecture
3. Train for 10 epochs with validation
4. Print training and validation metrics
5. Evaluate on the dev set
6. Save weights and vocabularies
7. Run inference on the test set

## Expected Performance

The model should achieve approximately:
- **Action accuracy**: 75-85%
- **Deprel accuracy**: 75-85%

## Loading a Trained Model

To load a previously trained model:

```python
from src.model_utils import load_model

# Load model and dictionaries
model, dictionaries = load_model(
    weights_path="models/parser_mlp_weights.h5",
    config_path="models/model_config.pkl",
    dictionaries_path="models/dictionaries.pkl"
)

# Extract dictionaries
form2id = dictionaries['form2id']
upos2id = dictionaries['upos2id']
id2action = dictionaries['id2action']
id2deprel = dictionaries['id2deprel']

# Use for inference
parsed_trees = model.run(
    test_trees, arc_eager, form2id, upos2id, id2action, id2deprel
)
```

## Model Architecture Details

```
Model: "parser_mlp"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_words (InputLayer)        [(None, 4)]          0                                            
__________________________________________________________________________________________________
input_pos (InputLayer)          [(None, 4)]          0                                            
__________________________________________________________________________________________________
embedding_words (Embedding)     (None, 4, 100)       vocab_size                                   
__________________________________________________________________________________________________
embedding_pos (Embedding)       (None, 4, 25)        upos_size                                    
__________________________________________________________________________________________________
flatten_words (Flatten)         (None, 400)          0                                            
__________________________________________________________________________________________________
flatten_pos (Flatten)           (None, 100)          0                                            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 500)          0                                            
__________________________________________________________________________________________________
hidden1 (Dense)                 (None, 64)           32064                                        
__________________________________________________________________________________________________
hidden2 (Dense)                 (None, 64)           4160                                         
__________________________________________________________________________________________________
output_action (Dense)           (None, 4)            260                                          
__________________________________________________________________________________________________
output_deprel (Dense)           (None, n_deprels)    varies                                       
==================================================================================================
```

## Troubleshooting

If you encounter memory issues:
- Reduce `batch_size` in the model initialization
- Reduce embedding dimensions (`word_emb_dim`, `pos_emb_dim`)
- Reduce `hidden_dim`

If training is too slow:
- Reduce `epochs`
- Increase `batch_size` (if memory allows)

If accuracy is too low:
- Increase `epochs`
- Increase `hidden_dim`
- Add more hidden layers (modify `_build_model()` in `src/model.py`)
- Increase embedding dimensions
