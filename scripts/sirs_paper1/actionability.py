"""
Actionability scorer: detects how much a claim pushes people to act.
Looks for imperative verbs, calls to action, panic-inducing language.
"""

import re


# Weighted keyword categories
ACTION_PATTERNS = {
    # Direct calls to action (highest weight)
    "call_to_action": {
        "weight": 1.0,
        "patterns": [
            r"\bappel(?:le|ons|ez)?\s+[àa]\b", r"\bboycott", r"\bmanifest",
            r"\bdescend(?:re|ez|ons)?\s+dans\s+la\s+rue\b", r"\bmobilis",
            r"\brefus(?:er|ez|ons)?\b", r"\brévolte", r"\bsoulève",
            r"\bdésobéi", r"\brésist", r"\bbloque[rz]?\b",
            r"\bn'achet[ez]+\s+p(?:as|lus)\b", r"\bne\s+vaccin",
            r"\bfuyez\b", r"\bévacue[rz]?\b", r"\bgrève\s+générale\b",
        ],
    },
    # Panic and urgency (high weight)
    "panic": {
        "weight": 0.8,
        "patterns": [
            r"\burgent\b", r"\balerte\b", r"\bdanger", r"\bmenace\b",
            r"\bpanique\b", r"\bfuyez\b", r"\bévacuation\b",
            r"\bcatastrophe\b", r"\bcrise\b", r"\battaque\b", r"\battentat\b",
            r"\bépidémie\b", r"\bcontamina", r"\bempoisonn",
            r"\bguerre\b", r"\baffrontement\b", r"\bémeute\b",
            r"\bmort[s]?\b", r"\btué[s]?\b", r"\bvictime[s]?\b",
        ],
    },
    # Emotional triggers (medium weight)
    "emotional": {
        "weight": 0.5,
        "patterns": [
            r"\bscandale\b", r"\bhonte\b", r"\btrahison\b", r"\bcorruption\b",
            r"\binjustice\b", r"\brévolt", r"\bindigna", r"\bcolère\b",
            r"\bintolér", r"\binsupportable\b", r"\binacceptable\b",
            r"\bcomplot\b", r"\bcoup d'[eé]tat\b", r"\bsecret",
        ],
    },
    # Imperative / directive language (medium weight)
    "directive": {
        "weight": 0.6,
        "patterns": [
            r"\bil\s+faut\b", r"\bon\s+doit\b", r"\bnous\s+devons\b",
            r"\bpartagez\b", r"\bdiffusez\b", r"\bfaites\s+attention\b",
            r"\bméfiez-vous\b", r"\bprotégez\b", r"\bpréparez-vous\b",
            r"\barmez-vous\b",
        ],
    },
}


class ActionabilityScorer:
    def __init__(self):
        self.compiled_patterns = {}
        for category, config in ACTION_PATTERNS.items():
            self.compiled_patterns[category] = {
                "weight": config["weight"],
                "patterns": [re.compile(p, re.IGNORECASE) for p in config["patterns"]],
            }

    def score(self, text):
        """
        Score the actionability of a text [0, 1].
        Higher = more likely to push people to act.
        """
        total_score = 0.0
        max_possible = 0.0

        for category, config in self.compiled_patterns.items():
            weight = config["weight"]
            matched = sum(1 for p in config["patterns"] if p.search(text))
            total_patterns = len(config["patterns"])

            if total_patterns > 0:
                category_score = (matched / total_patterns) * weight
                total_score += category_score
                max_possible += weight

        if max_possible == 0:
            return 0.0

        # Normalize and apply sigmoid-like curve
        raw = total_score / max_possible
        # Boost: even a few action words should push the score up
        boosted = min(1.0, raw * 3.5)
        return round(boosted, 4)
