import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
import os
import time
from collections import defaultdict
import random
import datetime
import numpy as np 
import sklearn.neighbors 
import json

#### Deep Metric Learning (DML) Imports
from pytorch_metric_learning import losses 
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator 

##### CONFIGURATION 
CURRENT_DATE_STR = datetime.datetime.now().strftime("%Y-%m-%d")

##### PATHS
DATA_DIR = "/fs/scratch/PAS3162/Kannally_Hargis/" 
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'unified_reid_data')
MODEL_SAVE_DIR = "/fs/ess/PAS3162/Kannally_Hargis_ess/trainedmodel/"
MODEL_FILENAME = f"{CURRENT_DATE_STR}_resnet50_arcface_dml_384.pth" 
FULL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
CANDIDATES_OUTPUT_PATH = os.path.join(MODEL_SAVE_DIR, "candidates_for_superglue.json")

# HYPERPARAMETERS
IMG_SIZE = 384                  
EMBEDDING_DIM = 512             
NUM_EPOCHS = 10
BATCH_SIZE = 32                 
BASE_LR = 3.5e-4
ARCFACE_LR = 5e-4               
SPLIT_RATIO = 0.8
OPEN_SET_THRESHOLD = 0.45       

#DEBUG MODE FLAG
DEBUG_MODE = False             
DEBUG_TOTAL_CAP = 1000          

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ROBUST DATA LOADING (UPDATED)
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading dataset...")
full_dataset = datasets.ImageFolder(TRAIN_DATA_PATH, transform=train_transforms)

class_indices = defaultdict(list)
for index, (_, label) in enumerate(full_dataset.samples):
    class_indices[label].append(index)

train_indices = []
test_indices = []
singleton_count = 0

random.seed(67) 

print("Splitting data...")
current_sample_count = 0

for label, indices in class_indices.items():
    if DEBUG_MODE and current_sample_count >= DEBUG_TOTAL_CAP:
        break

    num_samples = len(indices)
    
    if num_samples < 2:
        singleton_count += 1
        continue

    random.shuffle(indices)
    
    num_train = int(SPLIT_RATIO * num_samples)
    if num_train < 1: num_train = 1
    if num_samples >= 2 and (num_samples - num_train) < 1:
        num_train = num_samples - 1

    train_indices.extend(indices[:num_train])
    test_indices.extend(indices[num_train:])
    
    current_sample_count += num_samples

print(f"Splitting complete.")
if DEBUG_MODE:
    print(f"  [DEBUG] Cap Limit: {DEBUG_TOTAL_CAP}")
    print(f"  [DEBUG] Actual Total: {current_sample_count}")

print(f"  - Training Samples: {len(train_indices)}")
print(f"  - Test Samples: {len(test_indices)}")
print(f"  - Singletons Skipped: {singleton_count}")

# Helper wrapper to apply specific transforms to subsets
class TransformedSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, label

train_subset = Subset(full_dataset, train_indices)
test_subset = Subset(full_dataset, test_indices)

#ARCFACE LABEL MAPPING
print("Mapping training labels for ArcFace...")
raw_train_labels = [full_dataset.targets[i] for i in train_indices]
unique_labels = sorted(list(set(raw_train_labels)))
label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
num_classes = len(unique_labels)
print(f"  - Unique Training Classes: {num_classes}")

train_loader = DataLoader(
    train_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,          
    num_workers=4,    
    pin_memory=True          
) 

test_loader = DataLoader(
    test_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True          
)

###### MODEL SETUP 
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, EMBEDDING_DIM)
model = model.to(DEVICE)

#### ARCFACE LOSS COMPONENTS
loss_func = losses.ArcFaceLoss(
    num_classes=num_classes,
    embedding_size=EMBEDDING_DIM,
    margin=28.6, 
    scale=64,
).to(DEVICE)

optimizer = optim.Adam(
    [
        {'params': model.parameters(), 'lr': BASE_LR},
        {'params': loss_func.parameters(), 'lr': ARCFACE_LR} 
    ]
)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#EVALUATION (CPU-OPTIMIZED TO FIX OOM)
def evaluate_metrics(model, train_loader, test_loader, device):
    """Calculates Rank-1, 5, 10, 20, 50 and mAP on CPU to save VRAM."""
    model.eval() 
    
    # build Gallery (Training Set)
    gallery_embeddings = []
    gallery_labels = []
    
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            emb = torch.nn.functional.normalize(model(inputs), p=2, dim=1)
            gallery_embeddings.append(emb.cpu())
            gallery_labels.append(labels.cpu())
    
    gallery_embeddings = torch.cat(gallery_embeddings).numpy()
    gallery_labels = torch.cat(gallery_labels).numpy()

    # build Query (Test Set)
    query_embeddings = []
    query_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            emb = torch.nn.functional.normalize(model(inputs), p=2, dim=1)
            query_embeddings.append(emb.cpu())
            query_labels.append(labels.cpu())
            
    query_embeddings = torch.cat(query_embeddings).numpy()
    query_labels = torch.cat(query_labels).numpy()
    
    # Compute Metrics
    gallery_size = len(gallery_labels)
    max_k = min(50, gallery_size)
    
    distances, indices = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_k, 
        metric='cosine',
        n_jobs=-1
    ).fit(gallery_embeddings).kneighbors(query_embeddings)

    # Manual Rank Calculation (Numpy/CPU)
    predicted_labels = gallery_labels[indices]
    true_labels = query_labels.reshape(-1, 1)
    matches = (predicted_labels == true_labels)
    
    def get_rank_acc(k):
        if k > max_k: return 0.0
        hits = np.any(matches[:, :k], axis=1)
        return np.mean(hits)

    ranks = {
        "rank_1": get_rank_acc(1),
        "rank_5": get_rank_acc(5),
        "rank_10": get_rank_acc(10),
        "rank_20": get_rank_acc(20),
        "rank_50": get_rank_acc(50),
    }

    # Wrapper for Pytorch Metric Learning
    def KnnWrapper(query, k, reference, ref_includes_query=True, is_distance_metric=False):
        if isinstance(k, torch.Tensor): k = k.item()
        k = int(k)
        current_k = min(k, max_k)
        
        indices_tensor = torch.from_numpy(indices[:, :current_k]).long()
        distances_tensor = torch.from_numpy(distances[:, :current_k])
        return distances_tensor, indices_tensor

    calculator = AccuracyCalculator(
        include=("mean_average_precision",), 
        k=max_k, 
        knn_func=KnnWrapper,
        device=torch.device("cpu") 
    )
    
    metrics = calculator.get_accuracy(
        query=query_embeddings,
        query_labels=query_labels,
        reference=gallery_embeddings,
        reference_labels=gallery_labels,
        ref_includes_query=False 
    )
    
    metrics.update(ranks)
    return metrics

##### SUPERGLUE PREP
def predict_and_save_candidates(model, train_subset, test_subset, threshold=0.5, top_k=5):
    print("\n" + "="*40)
    print(f"GENERATING CANDIDATES FOR SUPERGLUE (Threshold: {threshold})")
    print("="*40)
    
    model.eval()
    
    ##### SETUP ORDERED LOADERS (Shuffle=False is CRITICAL for mapping paths)
    gallery_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    query_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    ##### BUILD GALLERY (Known Individuals)
    print("Building Known Gallery (with paths)...")
    gallery_embeddings = []
    gallery_paths = []
    gallery_labels = []

    # Get paths from the underlying dataset
    # subset.indices maps to the original full_dataset indices
    for idx in train_subset.indices:
        path, label = full_dataset.samples[idx]
        gallery_paths.append(path)
        gallery_labels.append(label)
        
    # Extract Embeddings
    with torch.no_grad():
        for inputs, _ in gallery_loader:
            inputs = inputs.to(DEVICE)
            emb = torch.nn.functional.normalize(model(inputs), p=2, dim=1)
            gallery_embeddings.append(emb.cpu())
            
    gallery_embeddings = torch.cat(gallery_embeddings).numpy()
    
    ##### BUILD KNN
    print("Fitting KNN...")
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=top_k, metric='cosine')
    knn.fit(gallery_embeddings)
    
    ####c QUERY (Test Data)
    print("Querying with Test Data...")
    results_for_superglue = []
    
    query_paths = []
    # Get query paths
    for idx in test_subset.indices:
        path, label = full_dataset.samples[idx]
        query_paths.append(path)

    with torch.no_grad():
        for batch_idx, (inputs, true_labels) in enumerate(query_loader):
            inputs = inputs.to(DEVICE)
            emb = torch.nn.functional.normalize(model(inputs), p=2, dim=1)
            emb_np = emb.cpu().numpy()
            true_labels_np = true_labels.numpy()
            
            dists, indices = knn.kneighbors(emb_np)
            
            for i in range(len(emb_np)):
                # Global index in the query list
                global_idx = batch_idx * BATCH_SIZE + i
                query_path = query_paths[global_idx]
                
                candidates = []
                for k in range(top_k):
                    dist = float(dists[i][k])
                    gal_idx = indices[i][k]
                    
                    # RETRIEVE PATH AND LABEL
                    match_path = gallery_paths[gal_idx]
                    pred_id = int(gallery_labels[gal_idx])
                    
                    candidates.append({
                        "rank": k+1,
                        "gallery_id": pred_id,
                        "gallery_path": match_path,
                        "distance": dist
                    })
                
                best_dist = candidates[0]['distance']
                
                # Logic: If distance is too high, ArcFace thinks it's new.
                # We still pass it to LightGlue to verify if ArcFace is right or wrong.
                if best_dist > threshold:
                    predicted_class = "new_individual"
                else:
                    predicted_class = candidates[0]['gallery_id']

                entry = {
                    "query_index": global_idx,
                    "query_path": query_path,     
                    "true_id": int(true_labels_np[i]),
                    "predicted_id_arcface": predicted_class,
                    "best_distance": best_dist,
                    "candidates": candidates
                }
                results_for_superglue.append(entry)

    with open(CANDIDATES_OUTPUT_PATH, 'w') as f:
        json.dump(results_for_superglue, f, indent=4)
        
    print(f"\nSaved {len(results_for_superglue)} query results to:")
    print(f"  -> {CANDIDATES_OUTPUT_PATH}")
    return results_for_superglue

###### MAIN EXECUTION
if __name__ == '__main__':
    best_loss = float('inf') 
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print("Starting Training (ArcFace)...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        model.train()
        loss_func.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            mapped_labels = torch.tensor(
                [label_map[l.item()] for l in labels], 
                device=DEVICE
            )
            
            optimizer.zero_grad()
            embeddings = model(inputs)
            loss = loss_func(embeddings, mapped_labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(inputs)

        scheduler.step()
        epoch_loss = running_loss / len(train_indices)
        print(f'TRAIN Loss: {epoch_loss:.4f}')

        metrics = evaluate_metrics(model, train_loader, test_loader, DEVICE)
        print(f"EVAL mAP: {metrics['mean_average_precision']:.4f}")
        print(f"EVAL Acc: R-1: {metrics['rank_1']:.4f} | R-5: {metrics['rank_5']:.4f} | R-10: {metrics['rank_10']:.4f} | R-20: {metrics['rank_20']:.4f} | R-50: {metrics['rank_50']:.4f}")

        if epoch_loss < best_loss: 
            best_loss = epoch_loss
            torch.save(model.state_dict(), FULL_SAVE_PATH)
            
    print(f'Training complete in {(time.time() - start_time) // 60:.0f}m.')

    predict_and_save_candidates(model, train_loader, test_loader, threshold=OPEN_SET_THRESHOLD, top_k=5)