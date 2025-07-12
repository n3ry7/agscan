import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

# Configuration
VALIDATION_DIR = Path("validation_images")
MODEL_PATH = Path("maturity_classifier.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Since we don't have labels, we'll assume the order is:
# First 4 = Class 0, Next 4 = Class 1, etc.
CLASS_NAMES = ['1_dark_green', '2_yellowish_green', '3_light_orange', '4_ripe_orange']

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def validate_unlabeled_images(validation_dir, model_path):
    # Load classifier
    try:
        classifier = joblib.load(model_path)
        print(f"Classifier loaded with classes: {classifier.classes_}")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return

    # Initialize feature extractor
    extractor = FeatureExtractor().to(DEVICE)
    extractor.eval()

    # Get validation images in order
    val_images = sorted(validation_dir.glob('*.[jJpP][pPnN][gG]'))
    if len(val_images) != 16:
        print(f"Warning: Expected 16 images, found {len(val_images)}")
    
    # Storage for results
    pred_labels = []
    all_probs = []
    image_paths = []
    
    print("\nProcessing images...")
    for i, img_path in enumerate(tqdm(val_images)):
        try:
            # Assign assumed true label based on order (4 images per class)
            assumed_class = i // 4
            if assumed_class > 3:
                assumed_class = 3  # Safety check
            
            # Process image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # Extract features
            with torch.no_grad():
                features = extractor(image_tensor).cpu().numpy()
            
            # Predict
            pred_label = classifier.predict(features)[0]
            probabilities = classifier.predict_proba(features)[0]
            
            # Store results
            pred_labels.append(pred_label)
            all_probs.append(probabilities)
            image_paths.append(img_path)
            
            print(f"\nImage {i+1}: {img_path.name}")
            print(f"Assumed Class: {CLASS_NAMES[assumed_class]} ({assumed_class})")
            print(f"Predicted Class: {CLASS_NAMES[pred_label]} ({pred_label})")
            print("Class Probabilities:")
            for cls_idx, prob in enumerate(probabilities):
                print(f"  {CLASS_NAMES[cls_idx]}: {prob:.4f}")
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue
    
    # Generate class distribution report
    print("\n\nClassification Distribution:")
    print("="*60)
    for class_idx, class_name in enumerate(CLASS_NAMES):
        count = sum(1 for pred in pred_labels if pred == class_idx)
        print(f"{class_name}: {count}/4 images")
    
    # Visualize predictions
    plt.figure(figsize=(10, 5))
    for i in range(4):
        class_probs = [probs[i] for probs in all_probs]
        plt.plot(range(16), class_probs, label=CLASS_NAMES[i])
    
    plt.title("Classification Probabilities Across Images")
    plt.xlabel("Image Number (1-16)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.savefig('class_probabilities.png')
    plt.show()

if __name__ == "__main__":
    validate_unlabeled_images(VALIDATION_DIR, MODEL_PATH)
