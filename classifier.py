import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
DATASET_PATH = Path("/home/nery/desafio_agscan/maturity_classes_final")
FEATURE_DB_PATH = Path("/home/nery/desafio_agscan/feature_db.npz")
MODEL_PATH = Path("/home/nery/desafio_agscan/maturity_classifier.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define maturity class mapping
CLASS_MAPPING = {
    '1_dark_green': 0,
    '2_yellowish_green': 1,
    '3_light_orange': 2,
    '4_ripe_orange': 3
}

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset Class
class OrangeMaturityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load all images and labels
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name in CLASS_MAPPING:
                    label = CLASS_MAPPING[class_name]
                    for img_path in class_dir.glob('*.jpg'):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Feature Extractor Class
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Use a pretrained ResNet without the final layer
        self.model = models.resnet50(pretrained=True)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        
        # Freeze all layers
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

# Create feature database
def create_feature_database(dataset, batch_size=32):
    # Initialize model
    extractor = FeatureExtractor().to(DEVICE)
    extractor.eval()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Storage for features and labels
    all_features = []
    all_labels = []
    
    # Extract features
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(DEVICE)
            features = extractor(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Combine all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Save feature database
    np.savez(FEATURE_DB_PATH, features=all_features, labels=all_labels)
    
    print(f"Feature database created with {len(all_features)} samples")
    return all_features, all_labels

# Train classifier
def train_classifier(features, labels):
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Initialize classifiers
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # Train SVM
    print("Training SVM classifier...")
    svm.fit(X_train, y_train)
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    rf.fit(X_train, y_train)
    
    # Evaluate both models
    svm_pred = svm.predict(X_test)
    rf_pred = rf.predict(X_test)
    
    svm_acc = accuracy_score(y_test, svm_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    print(f"\nSVM Test Accuracy: {svm_acc:.4f}")
    print(f"Random Forest Test Accuracy: {rf_acc:.4f}")
    
    # Save the better performing model
    if svm_acc > rf_acc:
        print("Saving SVM classifier as it performed better")
        joblib.dump(svm, MODEL_PATH)
        return svm
    else:
        print("Saving Random Forest classifier as it performed better")
        joblib.dump(rf, MODEL_PATH)
        return rf

# Classify a new orange
def classify_orange(image_path, classifier, extractor):
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Extract features
    with torch.no_grad():
        features = extractor(image_tensor).cpu().numpy()
    
    # Predict maturity stage
    probabilities = classifier.predict_proba(features)[0]
    prediction = classifier.predict(features)[0]
    
    # Get class names
    class_names = list(CLASS_MAPPING.keys())
    predicted_class = class_names[prediction]
    
    # Create result dictionary
    result = {
        'prediction': predicted_class,
        'probabilities': {
            class_names[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        },
        'features': features
    }
    
    return result

# Visualize classification
def visualize_classification(image_path, result):
    # Load image
    image = Image.open(image_path)
    
    # Create plot
    plt.figure(figsize=(15, 5))
    
    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Predicted: {result['prediction']}")
    plt.axis('off')
    
    # Plot probabilities
    plt.subplot(1, 2, 2)
    classes = list(result['probabilities'].keys())
    probs = list(result['probabilities'].values())
    
    colors = ['#2e8b57', '#9acd32', '#ffa500', '#ff4500']
    bars = plt.bar(classes, probs, color=colors)
    plt.title('Maturity Probabilities')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    
    # Add probability values on bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{Path(image_path).stem}_classification.jpg")
    plt.show()

# Main execution flow
if __name__ == "__main__":
    # Step 1: Create dataset
    print("Loading dataset...")
    dataset = OrangeMaturityDataset(DATASET_PATH, transform=transform)
    print(f"Loaded {len(dataset)} images")
    
    # Step 2: Create feature database
    if not FEATURE_DB_PATH.exists():
        print("Creating feature database...")
        features, labels = create_feature_database(dataset)
    else:
        print("Loading existing feature database...")
        db = np.load(FEATURE_DB_PATH)
        features = db['features']
        labels = db['labels']
    
    # Step 3: Train classifier
    if not MODEL_PATH.exists():
        print("Training classifier...")
        classifier = train_classifier(features, labels)
    else:
        print("Loading existing classifier...")
        classifier = joblib.load(MODEL_PATH)
    
    # Initialize feature extractor for classification
    extractor = FeatureExtractor().to(DEVICE)
    extractor.eval()
    
    # Step 4: Classify a new orange
    test_image = Path("/home/nery/desafio_agscan/test_orange.jpg")  # Replace with your test image
    if test_image.exists():
        print(f"\nClassifying {test_image.name}...")
        result = classify_orange(test_image, classifier, extractor)
        
        print(f"\nPredicted maturity: {result['prediction']}")
        print("Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
        
        # Visualize results
        visualize_classification(test_image, result)
    else:
        print(f"Test image not found: {test_image}")
