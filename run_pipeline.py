
import os
from score_curation import ScoreCurationPipeline

# Root directory containing raw input datasets
root_data_path = 'raw_data'

# Dataset configuration
dataset_name = "utilitarian"          # Name of the dataset
feature_key = "embed_text"             # Feature field used for embedding
score_key = "bin_score"                # Original score field to be curated

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
)

# Run the full pipeline (diagnosis + curation)
outputs = pipeline.run()

# Retrieve pipeline outputs
curated_dataset = outputs["dataset"]    # Dataset with curated and augmented scores
report = outputs["report"]              # Diagnosis report (transition matrix, rare samples, etc.)


# Optionally save the curated dataset to disk
# curated_dataset.to_json(f"{dataset_name}_curated.json")