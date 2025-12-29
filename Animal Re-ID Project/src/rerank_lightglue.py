import torch
import json
import os
import time
import random
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd


######CONFIGURATION

INPUT_JSON = "candidates_top100.json"
OUTPUT_JSON = "final_results_top100_half.json"
CHECKPOINT_JSON = "checkpoint_top100.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FRACTION OF DATA TO USE (0.5 = 50%)
SAMPLE_RATIO = 0.5 


######## MODEL SETUP

print(f"Loading LightGlue on {DEVICE}...")
# Using ALIKED + LightGlue
extractor = ALIKED(max_num_keypoints=1024, detection_threshold=0.2).eval().to(DEVICE)
matcher = LightGlue(features='aliked').eval().to(DEVICE)

def run_geometric_verification(query_path, candidates):
    """
    Reranks a list of 100 candidates based on geometric matches.
    """
    reranked_candidates = []
    
    # load & Extract Query Features ONCE (Optimization)
    try:
        img_q = load_image(query_path).to(DEVICE)
        feats_q = extractor.extract(img_q)
    except Exception as e:
        return candidates # Fail safe

    for cand in candidates:
        try:
            cand_path = cand['gallery_path']
            
            # load Candidate Image
            img_c = load_image(cand_path).to(DEVICE)
            feats_c = extractor.extract(img_c)
            
            # match
            with torch.no_grad():
                matches01 = matcher({"image0": feats_q, "image1": feats_c})
                feats_q_clean, feats_c_clean, matches01_clean = [
                    rbd(x) for x in [feats_q, feats_c, matches01]
                ]
            
            # score
            score = len(matches01_clean["matches"])
            cand['geo_matches'] = score
            reranked_candidates.append(cand)
            
        except Exception as e:
            cand['geo_matches'] = 0
            reranked_candidates.append(cand)

    # sort by matches (Descending), then distance (Ascending)
    reranked_candidates.sort(key=lambda x: (x['geo_matches'], -x['distance']), reverse=True)
    return reranked_candidates

def check_rank(candidates, true_label, k):
    """Returns 1 if true_label is in top k candidates"""
    for c in candidates[:k]:
        if c['gallery_label'] == true_label:
            return 1
    return 0


# MAIN EXECUTION
print(f"Loading {INPUT_JSON}...")
with open(INPUT_JSON, 'r') as f:
    all_queries = json.load(f)

# DOWNSAMPLING 
total_available = len(all_queries)
target_size = int(total_available * SAMPLE_RATIO)
print(f"Dataset Size: {total_available}. Sampling {int(SAMPLE_RATIO*100)}%...")

random.seed(42) 
selected_queries = random.sample(all_queries, target_size)
print(f"Processing {len(selected_queries)} queries against Top-100 candidates.")

results = []
# Stats tracking
stats = {'base_r1': 0, 'base_r5': 0, 'base_r25': 0,
         'glue_r1': 0, 'glue_r5': 0, 'glue_r25': 0}
total = 0

start_time = time.time()

for i, query in enumerate(selected_queries):
    q_path = query['query_path']
    true_label = query['true_label']
    original_candidates = query['candidates'] # This list has 100 items now
    
    # baseline Accuracy
    stats['base_r1'] += check_rank(original_candidates, true_label, 1)
    stats['base_r5'] += check_rank(original_candidates, true_label, 5)
    stats['base_r25'] += check_rank(original_candidates, true_label, 25)
        
    # run LightGlue
    new_candidates = run_geometric_verification(q_path, original_candidates)
    
    # new Accuracy
    stats['glue_r1'] += check_rank(new_candidates, true_label, 1)
    stats['glue_r5'] += check_rank(new_candidates, true_label, 5)
    stats['glue_r25'] += check_rank(new_candidates, true_label, 25)
        
    total += 1
    query['candidates'] = new_candidates
    results.append(query)

    #### LOGGING (Every 50) 
    if total % 50 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / total
        eta_hours = (avg_time * (len(selected_queries) - total)) / 3600
        
        print(f"[Processed {total}/{len(selected_queries)}] ETA: {eta_hours:.1f}h")
        print(f"   Baseline  -> R1: {stats['base_r1']/total:.1%} | R5: {stats['base_r5']/total:.1%} | R25: {stats['base_r25']/total:.1%}")
        print(f"   LightGlue -> R1: {stats['glue_r1']/total:.1%} | R5: {stats['glue_r5']/total:.1%} | R25: {stats['glue_r25']/total:.1%}")
        print("-" * 60)

    #SAFETY CHECKPOINT (Every 500)
    if total % 500 == 0:
        with open(CHECKPOINT_JSON, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"   (Checkpoint saved)")

# FINAL SUMMARY
print("\n" + "="*40)
print(f"FINISHED in {(time.time() - start_time)/3600:.2f} hours.")
print("-" * 40)
print(f"BASELINE (ArcFace Top-100):")
print(f"  R1: {stats['base_r1']/total:.2%} | R5: {stats['base_r5']/total:.2%} | R25: {stats['base_r25']/total:.2%}")
print("-" * 40)
print(f"RE-RANKED (LightGlue):")
print(f"  R1: {stats['glue_r1']/total:.2%} | R5: {stats['glue_r5']/total:.2%} | R25: {stats['glue_r25']/total:.2%}")
print("="*40)

with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=4)