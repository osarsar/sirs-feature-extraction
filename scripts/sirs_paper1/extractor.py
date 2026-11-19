"""
Feature extraction pipeline: combines all feature extractors.
Takes raw text, returns a feature vector for the model.
"""

from .topic_classifier import TopicClassifier
from .actionability import ActionabilityScorer
from .audience_scope import AudienceScopeScorer
from .temporal import TemporalSensitivityScorer


class FeatureExtractor:
    def __init__(self, taxonomy_path=None):
        self.topic_clf = TopicClassifier(taxonomy_path)
        self.actionability = ActionabilityScorer()
        self.audience = AudienceScopeScorer()
        self.temporal = TemporalSensitivityScorer()

    def extract(self, text):
        """
        Extract all features from raw text.
        Returns a dict with all dimensions.
        """
        topic_name, topic_rank, topic_sensitivity, topic_confidence = self.topic_clf.classify(text)

        return {
            "topic": topic_name,
            "topic_rank": topic_rank,
            "topic_sensitivity": topic_sensitivity,
            "topic_confidence": topic_confidence,
            "actionability": self.actionability.score(text),
            "audience_scope": self.audience.score(text),
            "temporal_sensitivity": self.temporal.score(text),
        }

    def extract_numeric(self, text):
        """
        Extract only numeric features for the model.
        Returns a list of floats.
        """
        features = self.extract(text)
        return [
            features["topic_sensitivity"],
            features["actionability"],
            features["audience_scope"],
            features["temporal_sensitivity"],
        ]

    @staticmethod
    def feature_names():
        return [
            "topic_sensitivity",
            "actionability",
            "audience_scope",
            "temporal_sensitivity",
        ]
