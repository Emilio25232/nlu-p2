from src.conllu.conllu_token import Token
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 64, 
                 epochs: int = 1, batch_size: int = 64, 
                 vocab_size_form: int = None, vocab_size_upos: int = None,
                 n_actions: int = 4, n_deprels: int = None,
                 pos_emb_dim: int = 25, n_word_feats: int = 4, n_pos_feats: int = 4):
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
            vocab_size_form (int): Size of the form vocabulary.
            vocab_size_upos (int): Size of the UPOS vocabulary.
            n_actions (int): Number of possible actions (default 4: SHIFT, LA, RA, REDUCE).
            n_deprels (int): Number of dependency relations.
            pos_emb_dim (int): Dimensionality of POS embeddings.
            n_word_feats (int): Number of word features (stack + buffer).
            n_pos_feats (int): Number of POS features (stack + buffer).
        """
        # Store hyperparameters
        self.word_emb_dim = word_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.vocab_size_form = vocab_size_form
        self.vocab_size_upos = vocab_size_upos
        self.n_actions = n_actions
        self.n_deprels = n_deprels
        self.n_word_feats = n_word_feats
        self.n_pos_feats = n_pos_feats
        
        # Build the model architecture
        self.model = None
        if vocab_size_form is not None and vocab_size_upos is not None and n_deprels is not None:
            self._build_model()
    
    def _build_model(self):
        """
        Builds the neural network architecture using Keras Functional API.
        Creates a multi-task model with two outputs: action prediction and deprel prediction.
        """
        # Define Input layers
        input_words = layers.Input(shape=(self.n_word_feats,), dtype='int32', name='input_words')
        input_pos = layers.Input(shape=(self.n_pos_feats,), dtype='int32', name='input_pos')
        
        # Add Embedding layers
        # Word embeddings (FORM)
        embedding_words = layers.Embedding(
            input_dim=self.vocab_size_form,
            output_dim=self.word_emb_dim,
            mask_zero=True,
            name='embedding_words'
        )(input_words)
        
        # POS embeddings (UPOS)
        embedding_pos = layers.Embedding(
            input_dim=self.vocab_size_upos,
            output_dim=self.pos_emb_dim,
            mask_zero=True,
            name='embedding_pos'
        )(input_pos)
        
        # Flatten embeddings
        flatten_words = layers.Flatten(name='flatten_words')(embedding_words)
        flatten_pos = layers.Flatten(name='flatten_pos')(embedding_pos)
        
        # Concatenate embeddings
        concatenated = layers.Concatenate(name='concatenate')([flatten_words, flatten_pos])
        
        # Add Dense hidden layers with ReLU activation
        hidden1 = layers.Dense(self.hidden_dim, activation='relu', name='hidden1')(concatenated)
        hidden2 = layers.Dense(self.hidden_dim, activation='relu', name='hidden2')(hidden1)
        
        # Add softmax output for transition actions
        output_action = layers.Dense(self.n_actions, activation='softmax', name='output_action')(hidden2)
        
        # Add softmax output for dependency relations
        output_deprel = layers.Dense(self.n_deprels, activation='softmax', name='output_deprel')(hidden2)
        
        # Create the model
        self.model = keras.Model(
            inputs=[input_words, input_pos],
            outputs=[output_action, output_deprel],
            name='parser_mlp'
        )
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss={
                'output_action': 'sparse_categorical_crossentropy',
                'output_deprel': 'sparse_categorical_crossentropy'
            },
            metrics={
                'output_action': 'accuracy',
                'output_deprel': 'accuracy'
            }
        )
    
    def train(self, X_train_words, X_train_pos, y_train_action, y_train_deprel,
              X_dev_words, X_dev_pos, y_dev_action, y_dev_deprel):
        """
        Trains the MLP model using the provided training and development data.

        This method trains the Keras model using the preprocessed arrays.

        Parameters:
            X_train_words: Training word features (numpy array).
            X_train_pos: Training POS features (numpy array).
            y_train_action: Training action labels (numpy array).
            y_train_deprel: Training deprel labels (numpy array).
            X_dev_words: Development word features (numpy array).
            X_dev_pos: Development POS features (numpy array).
            y_dev_action: Development action labels (numpy array).
            y_dev_deprel: Development deprel labels (numpy array).
        """
        if self.model is None:
            raise ValueError("Model not built. Please initialize with vocabulary sizes.")
        
        # Handle -1 labels for deprel (SHIFT/REDUCE actions don't have deprel)
        # Create sample weights: 1.0 for valid deprel labels, 0.0 for -1 labels
        train_deprel_weights = (y_train_deprel >= 0).astype(np.float32)
        dev_deprel_weights = (y_dev_deprel >= 0).astype(np.float32)
        
        # Replace -1 with 0 to avoid errors (these will be ignored by sample_weight)
        y_train_deprel_masked = np.where(y_train_deprel >= 0, y_train_deprel, 0)
        y_dev_deprel_masked = np.where(y_dev_deprel >= 0, y_dev_deprel, 0)
        
        # Sample weights as list (matching the y output order: [action, deprel])
        train_sample_weights = [
            np.ones_like(y_train_action, dtype=np.float32),  # weights for action output
            train_deprel_weights  # weights for deprel output
        ]
        
        dev_sample_weights = [
            np.ones_like(y_dev_action, dtype=np.float32),  # weights for action output
            dev_deprel_weights  # weights for deprel output
        ]
        
        # Prepare validation data
        validation_data = (
            [X_dev_words, X_dev_pos],
            [y_dev_action, y_dev_deprel_masked],
            dev_sample_weights
        )
        
        # Train the model
        history = self.model.fit(
            [X_train_words, X_train_pos],
            [y_train_action, y_train_deprel_masked],
            sample_weight=train_sample_weights,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        return history

    def evaluate(self, X_words, X_pos, y_action, y_deprel):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            X_words: Word features (numpy array).
            X_pos: POS features (numpy array).
            y_action: Action labels (numpy array).
            y_deprel: Deprel labels (numpy array).
        
        Returns:
            Evaluation results including loss and accuracy for both outputs.
        """
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        # Handle -1 labels for deprel (SHIFT/REDUCE actions don't have deprel)
        deprel_weights = (y_deprel >= 0).astype(np.float32)
        y_deprel_masked = np.where(y_deprel >= 0, y_deprel, 0)
        
        # Sample weights as list (matching the y output order: [action, deprel])
        sample_weights = [
            np.ones_like(y_action, dtype=np.float32),  # weights for action output
            deprel_weights  # weights for deprel output
        ]
        
        results = self.model.evaluate(
            [X_words, X_pos],
            [y_action, y_deprel_masked],
            sample_weight=sample_weights,
            batch_size=self.batch_size,
            verbose=1
        )
        
        return results
    
    def predict_and_evaluate(self, X_words, X_pos, y_action, y_deprel):
        """
        Get predictions and compute detailed evaluation metrics.
        
        Parameters:
            X_words: Word features (numpy array).
            X_pos: POS features (numpy array).
            y_action: True action labels (numpy array).
            y_deprel: True deprel labels (numpy array).
        
        Returns:
            Dictionary with evaluation metrics including:
            - transition_accuracy: Accuracy of action predictions
            - deprel_accuracy: Accuracy of deprel predictions (on arc actions)
            - joint_accuracy: Joint accuracy of both
        """
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        # Get predictions
        predictions = self.model.predict(
            [X_words, X_pos],
            batch_size=self.batch_size,
            verbose=0
        )
        
        # predictions is a list: [action_probs, deprel_probs]
        action_probs, deprel_probs = predictions
        
        # Get predicted labels
        y_pred_action = np.argmax(action_probs, axis=-1)
        y_pred_deprel = np.argmax(deprel_probs, axis=-1)
        
        # Compute transition accuracy
        action_correct = (y_action == y_pred_action)
        transition_accuracy = np.mean(action_correct)
        
        # Compute deprel accuracy only on arc-creating actions
        arc_mask = (y_deprel >= 0)
        num_arc_samples = np.sum(arc_mask)
        
        if num_arc_samples > 0:
            deprel_correct = (y_deprel[arc_mask] == y_pred_deprel[arc_mask])
            deprel_accuracy = np.mean(deprel_correct)
            
            # Joint accuracy
            joint_correct = action_correct[arc_mask] & deprel_correct
            joint_accuracy = np.mean(joint_correct)
        else:
            deprel_accuracy = 0.0
            joint_accuracy = 0.0
        
        return {
            'transition_accuracy': float(transition_accuracy),
            'deprel_accuracy': float(deprel_accuracy),
            'joint_accuracy': float(joint_accuracy),
            'total_samples': len(y_action),
            'arc_samples': int(num_arc_samples),
            'predictions': {
                'actions': y_pred_action,
                'deprels': y_pred_deprel
            }
        }
    
    def run(self, sents: list[list['Token']], arc_eager, form2id, upos2id, id2action, id2deprel, 
            nbuffer_feats: int = 2, nstack_feats: int = 2):
        """
        Executes the model on a list of sentences to perform dependency parsing.

        This method implements the vertical processing of sentences, predicting parser 
        transitions for each token in the sentences.

        Parameters:
            sents (list[list[Token]]): A list of sentences, where each sentence is represented 
                                       as a list of Token objects.
            arc_eager: Instance of ArcEager algorithm for state management.
            form2id: Dictionary mapping forms to IDs.
            upos2id: Dictionary mapping UPOS tags to IDs.
            id2action: Dictionary mapping IDs to action strings.
            id2deprel: Dictionary mapping IDs to dependency relations.
            nbuffer_feats: Number of buffer features to extract.
            nstack_feats: Number of stack features to extract.
        
        Returns:
            list[list[Token]]: Parsed sentences with head and dep fields filled.
        """
        from src.algorithm import Sample, Transition
        from src.vocab import PAD, UNK
        
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        # 1. Initialize: Create the initial state for each sentence
        states = [arc_eager.create_initial_state(sent) for sent in sents]
        sent_indices = list(range(len(sents)))  # Track which sentence each state belongs to
        
        # Main parsing loop
        while states:
            # 2. Feature Representation: Convert states to features
            samples = [Sample(state, Transition(arc_eager.SHIFT)) for state in states]
            
            n_word_feats = nstack_feats + nbuffer_feats
            X_words = []
            X_pos = []
            
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
                
                word_ids = [form2id.get(w, unk_id_form) for w in words]
                pos_ids = [upos2id.get(p, pad_id_pos) for p in pos_tags]
                
                X_words.append(word_ids)
                X_pos.append(pos_ids)
            
            X_words = np.asarray(X_words, dtype='int32')
            X_pos = np.asarray(X_pos, dtype='int32')
            
            # 3. Model Prediction: Predict actions and deprels
            action_probs, deprel_probs = self.model.predict([X_words, X_pos], verbose=0)
            
            # 4. Transition Sorting and 5. Validation Check
            new_states = []
            new_sent_indices = []
            
            for i, state in enumerate(states):
                # Sort actions by likelihood (descending)
                action_order = np.argsort(-action_probs[i])
                
                # Try each action in order of likelihood
                transition_applied = False
                for action_idx in action_order:
                    action = id2action[action_idx]
                    
                    # Check if this action is valid
                    is_valid = False
                    if action == arc_eager.SHIFT:
                        is_valid = len(state.B) > 0
                    elif action == arc_eager.LA:
                        is_valid = arc_eager.LA_is_valid(state)
                    elif action == arc_eager.RA:
                        is_valid = arc_eager.RA_is_valid(state)
                    elif action == arc_eager.REDUCE:
                        is_valid = arc_eager.REDUCE_is_valid(state)
                    
                    if is_valid:
                        # For LA and RA, select the most likely dependency label
                        if action in (arc_eager.LA, arc_eager.RA):
                            deprel_idx = np.argmax(deprel_probs[i])
                            deprel = id2deprel[deprel_idx]
                            transition = Transition(action, deprel)
                        else:
                            transition = Transition(action)
                        
                        # 6. State Update: Apply the transition
                        arc_eager.apply_transition(state, transition)
                        transition_applied = True
                        break
                
                if not transition_applied:
                    # Fallback: apply SHIFT if possible
                    if len(state.B) > 0:
                        arc_eager.apply_transition(state, Transition(arc_eager.SHIFT))
                
                # 7. Final State Check: Keep states that haven't reached final state
                if not arc_eager.final_state(state):
                    new_states.append(state)
                    new_sent_indices.append(sent_indices[i])
            
            # 8. Iterative Process: Update states for next iteration
            states = new_states
            sent_indices = new_sent_indices
        
        # Return parsed sentences with arcs converted to head/dep annotations
        return sents


if __name__ == "__main__":
    
    model = ParserMLP()