import os
import statistics
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import load_model
from src.predict import (
    predict, list_images, crop_box,
    CROP_LEFT_RATIO, CROP_TOP_RATIO, CROP_RIGHT_RATIO, CROP_BOTTOM_RATIO
)
from PIL import Image, ImageDraw


def visualize_crop_box(image_path, output_path, left_ratio=0.0, top_ratio=0.0, right_ratio=0.5, bottom_ratio=1.0):
    """
    Visualize the crop box on an image and save it.
    This function is used only for evaluation purposes.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the visualization
        left_ratio, top_ratio, right_ratio, bottom_ratio: Crop parameters matching crop_box()
    """
    # Load original image
    img = Image.open(image_path).convert("RGB")
    
    # Get crop coordinates using the same crop_box function
    _, crop_coords = crop_box(img, left_ratio, top_ratio, right_ratio, bottom_ratio, return_coords=True)
    
    # Create a copy to draw on
    img_with_box = img.copy()
    draw = ImageDraw.Draw(img_with_box)
    
    # Draw rectangle (crop box) in red with thick lines
    width, height = img.size
    line_width = max(3, width // 200)  # Adaptive line width
    draw.rectangle([
        (crop_coords['left'], crop_coords['top']),
        (crop_coords['right'], crop_coords['bottom'])
    ], outline="red", width=line_width)
    
    # Save the visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_with_box.save(output_path)
    print(f"  ✓ Crop box visualization saved: {output_path}")


def evaluate_model(test_data_dir="logs/data/test_data"):
    """
    Evaluate model performance on test dataset.
    
    Args:
        test_data_dir: Path to test data directory containing pos/ and neg/ folders
    
    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, and F1-score
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load model
    print("\n[1/3] Loading model and prototypes...")
    model, preprocess, prototypes, classes, device = load_model()
    print(f"  ✓ Model loaded successfully on device: {device}")
    print(f"  ✓ Classes: {classes}")
    
    # Prepare test data paths
    print("\n[2/3] Loading test data...")
    pos_folder = os.path.join(test_data_dir, "pos")
    neg_folder = os.path.join(test_data_dir, "neg")
    
    pos_images = list_images(pos_folder)
    neg_images = list_images(neg_folder)
    
    print(f"  ✓ Found {len(pos_images)} positive samples (open door)")
    print(f"  ✓ Found {len(neg_images)} negative samples (closed door)")
    
    total_samples = len(pos_images) + len(neg_images)
    if total_samples == 0:
        print("  ✗ No test images found!")
        return None
    
    # Run predictions
    print("\n[3/3] Running predictions...")
    print("-"*60)
    
    # Track predictions
    true_positives = 0  # Correctly predicted open
    true_negatives = 0  # Correctly predicted closed
    false_positives = 0  # Predicted open but actually closed
    false_negatives = 0  # Predicted closed but actually open
    
    results = []
    confidence_values = []  # Track all confidence values
    
    # Test positive samples (open door)
    print("\nTesting POSITIVE samples (open door):")
    for i, img_path in enumerate(pos_images, 1):
        img_name = os.path.basename(img_path)
        
        # For the first image, save visualizations
        if i == 1:
            # Visualize crop box (use same parameters as load_img preprocessing)
            viz_output_path = os.path.join("logs", "crop_box_visualization_pos.jpg")
            print(f"\n  Visualizing crop box for first positive image...")
            visualize_crop_box(img_path, viz_output_path, 
                             left_ratio=CROP_LEFT_RATIO, top_ratio=CROP_TOP_RATIO, 
                             right_ratio=CROP_RIGHT_RATIO, bottom_ratio=CROP_BOTTOM_RATIO)
            
            # Save edge-detected image
            edge_output_path = os.path.join("logs", "edge_detected_pos.jpg")
            result = predict(img_path, model, preprocess, prototypes, device, classes, save_processed_path=edge_output_path)
            print()
        else:
            result = predict(img_path, model, preprocess, prototypes, device, classes)
        
        # Combine closed and closed_dark as "closed" state
        closed_prob = result.get('closed', 0) + result.get('closed_dark', 0)
        open_prob = result.get('open', 0)
        
        # Predicted class
        predicted_class = "open" if open_prob > closed_prob else "closed"
        confidence = open_prob if predicted_class == "open" else closed_prob
        
        # Check if correct
        is_correct = (predicted_class == "open")
        if is_correct:
            true_positives += 1
            status = "✓"
        else:
            false_negatives += 1
            status = "✗"
        
        print(f"  {status} [{i}/{len(pos_images)}] {img_name:25} | Predicted: {predicted_class:6} | Confidence: {confidence:.2%} | Open: {open_prob:.2%}, Closed: {closed_prob:.2%}")
        
        confidence_values.append(confidence)
        
        results.append({
            'image': img_name,
            'true_label': 'open',
            'predicted_label': predicted_class,
            'confidence': confidence,
            'open_prob': open_prob,
            'closed_prob': closed_prob,
            'correct': is_correct
        })
    
    # Test negative samples (closed door)
    print("\nTesting NEGATIVE samples (closed door):")
    for i, img_path in enumerate(neg_images, 1):
        img_name = os.path.basename(img_path)
        
        # For the first image, save visualizations
        if i == 1:
            # Visualize crop box (use same parameters as load_img preprocessing)
            viz_output_path = os.path.join("logs", "crop_box_visualization_neg.jpg")
            print(f"\n  Visualizing crop box for first negative image...")
            visualize_crop_box(img_path, viz_output_path,
                             left_ratio=CROP_LEFT_RATIO, top_ratio=CROP_TOP_RATIO, 
                             right_ratio=CROP_RIGHT_RATIO, bottom_ratio=CROP_BOTTOM_RATIO)
            
            # Save edge-detected image
            edge_output_path = os.path.join("logs", "edge_detected_neg.jpg")
            result = predict(img_path, model, preprocess, prototypes, device, classes, save_processed_path=edge_output_path)
            print()
        else:
            result = predict(img_path, model, preprocess, prototypes, device, classes)
        
        # Combine closed and closed_dark as "closed" state
        closed_prob = result.get('closed', 0) + result.get('closed_dark', 0)
        open_prob = result.get('open', 0)
        
        # Predicted class
        predicted_class = "open" if open_prob > closed_prob else "closed"
        confidence = open_prob if predicted_class == "open" else closed_prob
        
        # Check if correct
        is_correct = (predicted_class == "closed")
        if is_correct:
            true_negatives += 1
            status = "✓"
        else:
            false_positives += 1
            status = "✗"
        
        print(f"  {status} [{i}/{len(neg_images)}] {img_name:25} | Predicted: {predicted_class:6} | Confidence: {confidence:.2%} | Open: {open_prob:.2%}, Closed: {closed_prob:.2%}")
        
        confidence_values.append(confidence)
        
        results.append({
            'image': img_name,
            'true_label': 'closed',
            'predicted_label': predicted_class,
            'confidence': confidence,
            'open_prob': open_prob,
            'closed_prob': closed_prob,
            'correct': is_correct
        })
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Accuracy: (TP + TN) / Total
    accuracy = (true_positives + true_negatives) / total_samples
    
    # Precision: TP / (TP + FP) - Of all predicted positives, how many are correct?
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # Recall (Sensitivity): TP / (TP + FN) - Of all actual positives, how many did we find?
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # F1-Score: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Specificity: TN / (TN + FP) - Of all actual negatives, how many did we correctly identify?
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    
    # Confidence metrics
    avg_confidence = statistics.mean(confidence_values) if confidence_values else 0.0
    median_confidence = statistics.median(confidence_values) if confidence_values else 0.0
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                Open      Closed")
    print(f"Actual Open     {true_positives:4d}      {false_negatives:4d}")
    print(f"       Closed   {false_positives:4d}      {true_negatives:4d}")
    
    print(f"\nBasic Metrics:")
    print(f"  Total Samples:     {total_samples}")
    print(f"  Correct:           {true_positives + true_negatives}")
    print(f"  Incorrect:         {false_positives + false_negatives}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:          {accuracy:.2%}  (Overall correctness)")
    print(f"  Precision:         {precision:.2%}  (When predicting 'open', how often correct)")
    print(f"  Recall:            {recall:.2%}  (Of all 'open' doors, how many detected)")
    print(f"  F1-Score:          {f1_score:.2%}  (Harmonic mean of precision & recall)")
    print(f"  Specificity:       {specificity:.2%}  (Of all 'closed' doors, how many detected)")
    
    print(f"\nConfidence Metrics:")
    print(f"  Average Confidence: {avg_confidence:.2%}  (Mean confidence across all predictions)")
    print(f"  Median Confidence:  {median_confidence:.2%}  (Middle confidence value)")
    
    print("\nInterpretation:")
    if accuracy >= 0.95:
        print(f"  ✓ Excellent accuracy! Model performs very well.")
    elif accuracy >= 0.85:
        print(f"  ✓ Good accuracy. Model performs well overall.")
    elif accuracy >= 0.70:
        print(f"  ⚠ Fair accuracy. Model may need improvement.")
    else:
        print(f"  ✗ Poor accuracy. Model needs significant improvement.")
    
    if recall < 0.80:
        print(f"  ⚠ Low recall: Model misses {(1-recall)*100:.0f}% of open doors (false negatives)")
    
    if precision < 0.80:
        print(f"  ⚠ Low precision: {(1-precision)*100:.0f}% of 'open' predictions are false alarms")
    
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'avg_confidence': avg_confidence,
        'median_confidence': median_confidence,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_samples': total_samples,
        'results': results
    }


if __name__ == "__main__":
    # Run evaluation
    metrics = evaluate_model()
    
    if metrics:
        print("\n✓ Evaluation complete!")
    else:
        print("\n✗ Evaluation failed!")

