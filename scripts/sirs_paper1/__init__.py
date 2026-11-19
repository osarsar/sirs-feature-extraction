"""Original Paper 1 (SIRS) rule-based extractors, copied unchanged from
societal-risk-scorer/src/features (only import paths adapted)."""
from .extractor import FeatureExtractor

# Paper 1 formula weights (Sarsar et al., SIRS, Sec. 3.2.1; configs/config.yaml)
PAPER1_WEIGHTS = [0.20, 0.10, 0.55, 0.15]
