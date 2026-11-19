"""
Topic classifier: maps a claim text to one of 21 Moroccan media topics.
Uses keyword matching with TF-IDF-like scoring.
"""

import json
import os
import re
from collections import defaultdict


class TopicClassifier:
    def __init__(self, taxonomy_path=None):
        if taxonomy_path is None:
            taxonomy_path = os.path.join(os.path.dirname(__file__), "topic_taxonomy.json")
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)

        self.topics = taxonomy["topics"]
        self.topic_keywords = {}
        self.topic_sensitivity = {}
        self.topic_rank = {}

        for t in self.topics:
            name = t["name"]
            self.topic_keywords[name] = [kw.lower() for kw in t["keywords"]]
            self.topic_sensitivity[name] = t["sensitivity"]
            self.topic_rank[name] = t["rank"]

    def classify(self, text):
        """
        Classify text into a topic. Returns (topic_name, topic_rank, topic_sensitivity, confidence).
        """
        text_lower = text.lower()
        scores = {}

        for topic, keywords in self.topic_keywords.items():
            score = 0
            matched = 0
            for kw in keywords:
                # Multi-word keywords get higher weight
                weight = len(kw.split())
                if kw in text_lower:
                    score += weight
                    matched += 1

            if matched > 0:
                # Normalize by total keywords to favor specificity
                scores[topic] = score * (matched / len(keywords))

        if not scores:
            # Default to lowest sensitivity topic
            return "Coupe du monde", 21, 0.05, 0.0

        best_topic = max(scores, key=scores.get)
        max_score = scores[best_topic]
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0

        return (
            best_topic,
            self.topic_rank[best_topic],
            self.topic_sensitivity[best_topic],
            round(confidence, 4),
        )

    def get_sensitivity(self, topic_name):
        return self.topic_sensitivity.get(topic_name, 0.05)

    def get_rank(self, topic_name):
        return self.topic_rank.get(topic_name, 21)
