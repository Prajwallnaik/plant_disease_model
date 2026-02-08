import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = 300 # EfficientNetB3 input size
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "model/efficientnet_b3_tomato.pth"

print(f"Using device: {DEVICE}")

# Data Transforms (Augmentation for train, normalization for all)
# ImageNet normalization
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    NORMALIZE
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    NORMALIZE
])

# Load Datasets
print("Loading Datasets...")
# Check if data exists
if not os.path.exists("data/train"):
    print("Error: data/train not found.")
    exit(1)

train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
val_dataset = datasets.ImageFolder("data/val", transform=val_transforms)
test_dataset = datasets.ImageFolder("data/test", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows compatibility safety
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_dataset.classes
print(f"Classes: {class_names}")

# Build Model
print("Building Model (EfficientNetB3)...")
# Weights=IMAGENET1K_V1 is the modern way to load pretrained weights
try:
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
except:
    # Fallback for older torchvision if needed, though 3.14 should have latest
    model = models.efficientnet_b3(pretrained=True)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Modify Classifier
# EfficientNet's classifier is a Sequential block. We replace it.
# Inspecting structure: model.classifier[1] is the Linear layer
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_features, len(class_names))
)

model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train_model():
    print("Starting Training...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"[{time.time() - start_time:.0f}s] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  -> Saved best model")

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.4f}")

# Train and Evaluate
if __name__ == "__main__":
    train_model()
    
    # Final Evaluation on Test Set
    if os.path.exists(MODEL_SAVE_PATH):
        print("\nEvaluating on Test Set...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        print(f"Test Accuracy: {test_correct/test_total:.4f}")
    else:
        print("Model file not found, skipping evaluation.")
