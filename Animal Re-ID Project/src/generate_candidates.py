import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import sklearn.neighbors
import json
import os
import sys
from collections import defaultdictcandidates_top100
import random


WEIGHTS_PATH = "/fs/ess/PAS3162/Kannally_Hargis_ess/trainedmodel/MiewID_ArcFace_FineTun.pth"
DATA_DIR = "/fs/scratch/PAS3162/Kannally_Hargis/unified_reid_data"
OUTPUT_JSON_PATH = "candidates_top100.json"

BATCH_SIZE = 32
IMG_SIZE = 384      
EMBEDDING_DIM = 512  
TOP_K = 100          # We want Top 100 for LightGlue

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

print("Setting up data...")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# re-create the exact split using seed 67
print("Re-creating Train/Test split...")
class_indices = defaultdict(list)
for index, (_, label) in enumerate(full_dataset.samples):
    class_indices[label].append(index)

train_indices = []
test_indices = []
random.seed(67) # match the training seed

for label, indices in class_indices.items():
    if len(indices) < 2: continue 
    random.shuffle(indices)
    num_train = int(0.8 * len(indices))
    if num_train < 1: num_train = 1
    if len(indices) >= 2 and (len(indices) - num_train) < 1: num_train = len(indices) - 1
    
    train_indices.extend(indices[:num_train])
    test_indices.extend(indices[num_train:])

train_subset = Subset(full_dataset, train_indices)
test_subset = Subset(full_dataset, test_indices)

print(f"Gallery Size (Train): {len(train_subset)}")
print(f"Query Size (Test): {len(test_subset)}")


###### SETUP MODEL 

print("Setting up MiewID architecture...")
model = models.resnet50(weights=None) # Random init

# output 512 embeddings, NOT class probabilities
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, EMBEDDING_DIM) 

model = model.to(DEVICE)

# load Weights
print(f"Loading weights from {WEIGHTS_PATH}...")
if os.path.exists(WEIGHTS_PATH):
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(clean_state_dict, strict=True)
        print("MiewID Weights loaded successfully!")
    except Exception as e:
        print(f"Weight loading error: {e}")
        print("Trying non-strict load...")
        model.load_state_dict(clean_state_dict, strict=False)
    
    model.eval()
else:
    print(f"❌ ERROR: File not found at {WEIGHTS_PATH}")
    sys.exit(1)


#### GENERATE CANDIDATES

print(f"Generating Top {TOP_K} Candidates...")

gallery_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
query_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Embed Gallery
gallery_embeddings = []
gallery_paths = []
gallery_labels = []

# Map paths
for idx in train_subset.indices:
    path, label = full_dataset.samples[idx]
    gallery_paths.append(path)
    gallery_labels.append(label)

print("Embedding Gallery...")
with torch.no_grad():
    for inputs, _ in gallery_loader:
        inputs = inputs.to(DEVICE)
        # Normalize embeddings, ArcFace needs normalized vectors
        emb = torch.nn.functional.normalize(model(inputs), p=2, dim=1)
        gallery_embeddings.append(emb.cpu())
gallery_embeddings = torch.cat(gallery_embeddings).numpy()

#fit KNN
print(f"Fitting KNN on {len(gallery_embeddings)} items...")
knn = sklearn.neighbors.NearestNeighbors(n_neighbors=TOP_K, metric='cosine')
knn.fit(gallery_embeddings)

#Query
print("Querying Test Set...")
results = []
query_paths = []

for idx in test_subset.indices:
    path, label = full_dataset.samples[idx]
    query_paths.append(path)

with torch.no_grad():
    for batch_idx, (inputs, true_labels) in enumerate(query_loader):
        inputs = inputs.to(DEVICE)
        # normalize query
        emb = torch.nn.functional.normalize(model(inputs), p=2, dim=1)
        emb_np = emb.cpu().numpy()
        true_labels_np = true_labels.numpy()
        
        dists, indices = knn.kneighbors(emb_np)
        
        for i in range(len(emb_np)):
            global_idx = batch_idx * BATCH_SIZE + i
            query_path = query_paths[global_idx]
            
            candidates = []
            for k in range(TOP_K):
                dist = float(dists[i][k])
                gal_idx = indices[i][k]
                
                candidates.append({
                    "rank": k+1,
                    "gallery_path": gallery_paths[gal_idx],
                    "gallery_label": int(gallery_labels[gal_idx]),
                    "distance": dist
                })
            
            results.append({
                "query_path": query_path,
                "true_label": int(true_labels_np[i]),
                "candidates": candidates
            })
        
        if batch_idx % 50 == 0:
            print(f"Processed batch {batch_idx}...")

# save json
with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n✅ Done! Saved candidates to {OUTPUT_JSON_PATH}")