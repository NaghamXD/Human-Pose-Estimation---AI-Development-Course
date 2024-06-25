
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


### Add Wandb in order to show results there  || or in tensorboard

def precision_recall_curve(gt_boxes: list, 
                           pred_boxes: list, 
                           scores: list, iou_threshold=0.5):
    """ Compute and visualize precision-recall curve. """
    all_tp, all_fp, all_fn = 0, 0, 0
    
    # Sort predictions by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    pred_boxes_list = [pred_boxes_list[i] for i in sorted_indices]
    scores_list = [scores[i] for i in sorted_indices]
    
    precisions = []
    recalls = []
    
    for threshold in scores_list:
        tp, fp, fn = 0, 0, 0
        for gt_boxes, pred_boxes in zip(gt_boxes, pred_boxes_list):
            # Filter predictions by current threshold
            pred_boxes_thr = [box for box, score in zip(pred_boxes, scores_list) if score >= threshold]
            t, f_p, f_n = match_boxes(gt_boxes, pred_boxes_thr, iou_threshold)
            tp += t
            fp += f_p
            fn += f_n
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
    
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()