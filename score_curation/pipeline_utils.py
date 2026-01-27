
from score_curation import run_diagnose
from score_curation import run_curation


class ScoreCurationPipeline:
    """
    End-to-end pipeline for recommendation system score diagnosis and curation.

    The pipeline consists of two stages:
      1) Diagnosis: detect mislabeled samples and rare patterns
      2) Curation: revise scores based on confidence and diversity signals

    Inputs:
      - Raw dataset (JSON / JSONL / Parquet)
      - Feature column used for embedding
      - Original score column to be curated

    Outputs:
      - Curated dataset with additional score fields
      - Diagnostic report containing transition matrices and error statistics
    """

    def __init__(
        self,
        config_path: str,
        dataset_name: str,
        dataset_path: str,
        feature_key: str = "embed_text",
        score_key: str = "bin_score",
        output_dir: str = "results",
    ):
        """
        Initialize the score curation pipeline.

        Args:
            config_path: Path to the system / model configuration file
            dataset_name: Dataset identifier (used for naming outputs)
            dataset_path: Path to the raw dataset file
            feature_key: Column name for textual features used in embedding
            score_key: Column name for the original (noisy) scores
            output_dir: Directory to store reports and curated datasets
        """
        self.config_path = config_path
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.feature_key = feature_key
        self.score_key = score_key
        self.output_dir = output_dir

        # Load base configuration
        from docta.utils.config import Config
        self.cfg = Config.fromfile(config_path)

        # Inject runtime-specific parameters into the config
        # (These override static config values)
        self.cfg.dataset_type = dataset_name
        self.cfg.feature_key = feature_key
        self.cfg.score_key = score_key
        self.cfg.save_path = output_dir

    # -----------------------------
    # Stage 1: Diagnosis
    # -----------------------------
    def run_diagnosis(self):
        """
        Run the diagnosis stage.

        This stage:
          - Loads the raw dataset
          - Extracts embeddings
          - Detects score inconsistencies and rare patterns
          - Saves a diagnosis report to disk

        Returns:
            report: Diagnosis report object
        """

        print("[Pipeline] Running diagnosis...")
        self.report = run_diagnose(
            cfg=self.cfg,
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            output_dir=self.output_dir,
            feature_key=self.feature_key,
            score_key = self.score_key,
        )
        return self.report

    # -----------------------------
    # Stage 2: Score Curation
    # -----------------------------
    def run_curation(self):
        """
        Run the score curation stage.

        This stage:
          - Loads the diagnosis report
          - Revises noisy scores based on confidence thresholds
          - Computes diversity-aware scores
          - Writes curated scores back into the dataset

        Returns:
            curated_dataset: Dataset augmented with curated scores
        """

        print("[Pipeline] Running score curation...")
        self.curated_dataset = run_curation(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            output_dir=self.output_dir,
            score_key=self.score_key,
            confidence_prob=0.5,  # Confidence threshold for score revision
        )
        return self.curated_dataset

    # -----------------------------
    # Full pipeline
    # -----------------------------
    def run(self):
        """
        Execute the full score curation pipeline.

        This method sequentially runs:
          1) Diagnosis
          2) Score curation

        Returns:
            dict with keys:
              - "dataset": curated dataset
              - "report": diagnosis report
        """
        self.run_diagnosis()
        self.run_curation()

        return {
            "dataset": self.curated_dataset,
            "report": self.report,
        }
