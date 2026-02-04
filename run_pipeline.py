
import os
from score_curation import ScoreCurationPipeline
from score_curation.data_preprocess import preprocess_dataset
import json

# Root directory containing raw input datasets
root_data_path = 'raw_data'

# # Dataset configuration
##############################
# dataset_name = "utilitarian"          # Name of the dataset
# feature_key = "embed_text"             # Feature field used for embedding
# score_key = "bin_score"                # Original score field to be curated
# num_classes = 2
# confidence_prob=0.5
# # Output directory for reports and curated datasets
# output_dir = 'results/'

##############################
# Dataset configuration
dataset_name = "timeline_label"          # Name of the dataset
num_classes = 6
confidence_prob=0.5
# Output directory for reports and curated datasets
output_dir = 'results'
org_score_key = "label_score"           # Original continuous score field
org_feature_keys=["history", "label_title"]


# ============================================================
# Step 0: Data Preprocessing (optional)
# - Discretize continuous scores into bins (bin_score)
# - Concatenate feature fields into embed_text
# ============================================================
# Uncomment below to enable preprocessing before pipeline
feature_key = "embed_text"             # Feature field used for embedding
score_key = "bin_score"                # Original score field to be curated
raw_input_path = os.path.join(root_data_path, f"{dataset_name}_raw.json")
processed_output_path = os.path.join(root_data_path, f"{dataset_name}.json")
preprocess_dataset(
    input_path=raw_input_path,
    output_path=processed_output_path,
    num_classes=num_classes,
    score_key=org_score_key,              # Original continuous score field
    feature_keys=org_feature_keys,  # Fields to concatenate
    output_score_key=score_key,           # Output: bin_score
    output_feature_key=feature_key,       # Output: embed_text
)

# ============================================================
# Initialize the score curation pipeline
# This pipeline runs:
#   1) score diagnosis (error detection + rare pattern detection)
#   2) score curation (confidence-based score correction + diversity scoring)
pipeline = ScoreCurationPipeline(
    config_path="template.py",                                # Model / system configuration
    dataset_name=dataset_name,                                # Dataset identifier
    dataset_path=processed_output_path,  # Path to raw dataset
    feature_key=feature_key,                                  # Feature column name
    score_key=score_key,                                      # Score column name
    output_dir=output_dir,                                    # Output directory
    num_classes=num_classes,
    confidence_prob=confidence_prob,
)

outputs = pipeline.run()

# Retrieve pipeline outputs
curated_dataset = outputs["dataset"]    # Dataset with curated and augmented scores
report = outputs["report"]              # Diagnosis report (transition matrix, rare samples, etc.)

# present mis-corrupted samples

selected_size=10
curated_samples_info = report.curation['score_curation']  # (idx, score, confidence)
idx_list = [int(item[0]) for item in curated_samples_info][:selected_size]
curated_subset = curated_dataset.select(idx_list)

for i, (idx, score, confidence) in enumerate(curated_samples_info[:selected_size]):
    print(f"\n==== sample {i} (orig_idx={int(idx)}) ====")
    print(f"bin_score={int(score)}, confidence={float(confidence):.2f}")

    print(json.dumps(curated_subset[i], indent=4, ensure_ascii=False))
    
# Optionally save the curated dataset to disk
# curated_dataset.to_json(f"{dataset_name}_curated.json")
