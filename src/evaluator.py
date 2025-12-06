"""
Evaluation utilities for the dependency parser.
"""

import numpy as np
from typing import Dict, Tuple


def evaluate_predictions(y_true_action: np.ndarray, 
                         y_pred_action: np.ndarray,
                         y_true_deprel: np.ndarray, 
                         y_pred_deprel: np.ndarray) -> Dict[str, float]:
    """
    Compute detailed evaluation metrics for parser predictions.
    
    Parameters:
        y_true_action: True action labels (numpy array).
        y_pred_action: Predicted action labels (numpy array).
        y_true_deprel: True deprel labels (numpy array, -1 for non-arc actions).
        y_pred_deprel: Predicted deprel labels (numpy array).
    
    Returns:
        Dictionary containing:
        - transition_accuracy: Overall transition accuracy
        - deprel_accuracy: Deprel accuracy (only on arc actions)
        - joint_accuracy: Joint accuracy (both transition and deprel correct)
        - total_samples: Total number of samples
        - arc_samples: Number of arc-creating actions (LEFT-ARC/RIGHT-ARC)
    """
    # Compute transition (action) accuracy
    action_correct = (y_true_action == y_pred_action)
    transition_accuracy = np.mean(action_correct)
    
    # Compute deprel accuracy only on arc-creating actions (where y_true_deprel >= 0)
    arc_mask = (y_true_deprel >= 0)
    num_arc_samples = np.sum(arc_mask)
    
    if num_arc_samples > 0:
        deprel_correct = (y_true_deprel[arc_mask] == y_pred_deprel[arc_mask])
        deprel_accuracy = np.mean(deprel_correct)
        
        # Joint accuracy: both action and deprel must be correct (only for arc actions)
        joint_correct = action_correct[arc_mask] & deprel_correct
        joint_accuracy = np.mean(joint_correct)
    else:
        deprel_accuracy = 0.0
        joint_accuracy = 0.0
    
    return {
        'transition_accuracy': float(transition_accuracy),
        'deprel_accuracy': float(deprel_accuracy),
        'joint_accuracy': float(joint_accuracy),
        'total_samples': len(y_true_action),
        'arc_samples': int(num_arc_samples),
        'non_arc_samples': int(len(y_true_action) - num_arc_samples)
    }


def evaluate_model_on_dev(model, X_dev_words, X_dev_pos, y_dev_action, y_dev_deprel, 
                          batch_size: int = 64) -> Dict[str, float]:
    """
    Evaluate model on development set with detailed metrics.
    
    Parameters:
        model: Trained ParserMLP model.
        X_dev_words: Development word features.
        X_dev_pos: Development POS features.
        y_dev_action: Development action labels.
        y_dev_deprel: Development deprel labels.
        batch_size: Batch size for prediction.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    # Get predictions from model
    predictions = model.model.predict(
        [X_dev_words, X_dev_pos],
        batch_size=batch_size,
        verbose=0
    )
    
    # predictions is a list: [action_probs, deprel_probs]
    action_probs, deprel_probs = predictions
    
    # Get predicted labels (argmax)
    y_pred_action = np.argmax(action_probs, axis=-1)
    y_pred_deprel = np.argmax(deprel_probs, axis=-1)
    
    # Compute detailed metrics
    metrics = evaluate_predictions(y_dev_action, y_pred_action, 
                                   y_dev_deprel, y_pred_deprel)
    
    return metrics


def print_evaluation_metrics(metrics: Dict[str, float], dataset_name: str = "Development"):
    """
    Pretty-print evaluation metrics.
    
    Parameters:
        metrics: Dictionary containing evaluation metrics.
        dataset_name: Name of the dataset (e.g., "Development", "Test").
    """
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name} Set Evaluation Metrics")
    print(f"{'='*60}")
    print(f"  Transition Accuracy:     {metrics['transition_accuracy']:.4f} ({metrics['transition_accuracy']*100:.2f}%)")
    print(f"  Deprel Accuracy:         {metrics['deprel_accuracy']:.4f} ({metrics['deprel_accuracy']*100:.2f}%)")
    print(f"  Joint Accuracy:          {metrics['joint_accuracy']:.4f} ({metrics['joint_accuracy']*100:.2f}%)")
    print(f"  Total Samples:           {metrics['total_samples']:,}")
    print(f"  Arc Actions:             {metrics['arc_samples']:,}")
    print(f"  Non-Arc Actions:         {metrics['non_arc_samples']:,}")
    print(f"{'='*60}\n")
