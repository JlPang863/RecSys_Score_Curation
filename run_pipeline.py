
import os
from score_curation import ScoreCurationPipeline
import json

# Root directory containing raw input datasets
root_data_path = 'raw_data'

# Dataset configuration
dataset_name = "utilitarian"          # Name of the dataset
feature_key = "embed_text"             # Feature field used for embedding
score_key = "bin_score"                # Original score field to be curated
num_classes = 2
# Output directory for reports and curated datasets
output_dir = 'results/'


# Initialize the score curation pipeline
# This pipeline runs:
#   1) score diagnosis (error detection + rare pattern detection)
#   2) score curation (confidence-based score correction + diversity scoring)
pipeline = ScoreCurationPipeline(
    config_path="template.py",                                # Model / system configuration
    dataset_name=dataset_name,                                # Dataset identifier
    dataset_path=os.path.join(root_data_path, f"{dataset_name}.json"),  # Path to raw dataset
    feature_key=feature_key,                                  # Feature column name
    score_key=score_key,                                      # Score column name
    output_dir=output_dir,                                    # Output directory
    num_classes=num_classes,
)

# Run the full pipeline (diagnosis + curation)
outputs = pipeline.run()

# Retrieve pipeline outputs
curated_dataset = outputs["dataset"]    # Dataset with curated and augmented scores
report = outputs["report"]              # Diagnosis report (transition matrix, rare samples, etc.)

# present mis-corrupted samples

selected_size=10
curated_samples_info = report.curation['score_curation']  # (idx, score, confidence)
idx_list = [int(item[0]) for item in curated_samples_info][:selected_size]
curated_subset = curated_dataset.select(idx_list)

# import pdb;pdb.set_trace()

curated_subset = curated_subset.remove_columns(["salient_labels", "id"])

for i, (idx, score, confidence) in enumerate(curated_samples_info[:selected_size]):
    print(f"\n==== sample {i} (orig_idx={int(idx)}) ====")
    print(f"bin_score={int(score)}, confidence={float(confidence):.2f}")

    print(json.dumps(curated_subset[i], indent=4, ensure_ascii=False))
    
# Optionally save the curated dataset to disk
# curated_dataset.to_json(f"{dataset_name}_curated.json")