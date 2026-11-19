"""
Temporal sensitivity detector: is the claim happening in a sensitive time context?
Detects crisis, election, religious events, urgency markers.
"""

import re


TEMPORAL_PATTERNS = {
    # Crisis / emergency (highest sensitivity)
    "crisis": {
        "weight": 1.0,
        "patterns": [
            r"\bcrise\b", r"\burgence\b", r"\bcatastrophe\b", r"\bséisme\b",
            r"\binondation\b", r"\bépidémie\b", r"\bpandémie\b",
            r"\bguerre\b", r"\bconflit\s+armé\b", r"\battentat\b",
            r"\bcouvre-feu\b", r"\bétat\s+d'urgence\b", r"\balerte\s+maximale\b",
            r"\bimmédiat", r"\bsur-le-champ\b", r"\ben\s+cours\b",
            r"\bcette\s+nuit\b", r"\baujourd'hui\b",
        ],
    },
    # Electoral period
    "electoral": {
        "weight": 0.9,
        "patterns": [
            r"\bélection", r"\bscrutin\b", r"\bvote\b", r"\bcampagne\s+électorale\b",
            r"\bcandidat", r"\bbureau(?:x)?\s+de\s+vote\b", r"\bjour\s+du\s+vote\b",
            r"\brésultats?\s+(?:du\s+)?(?:scrutin|élection)\b",
            r"\bdépouill", r"\bpré-élector",
        ],
    },
    # Religious sensitivity
    "religious": {
        "weight": 0.7,
        "patterns": [
            r"\bramadan\b", r"\baïd\b", r"\bmawlid\b", r"\bachoura\b",
            r"\bvendredi\b.*\bprière\b", r"\bprêche\b", r"\bmosquée\b",
            r"\blieu\s+saint\b", r"\bpèlerinage\b", r"\bsacré\b",
        ],
    },
    # Social tension
    "tension": {
        "weight": 0.8,
        "patterns": [
            r"\btension\b", r"\baffront", r"\bémeute\b", r"\bviolence\b",
            r"\bclash\b", r"\bincident\b", r"\bdébordement\b",
            r"\bmouvement\s+social\b", r"\bescalade\b", r"\brupture\b",
        ],
    },
    # Urgency markers
    "urgency": {
        "weight": 0.6,
        "patterns": [
            r"\bdemain\b", r"\bce\s+soir\b", r"\bdans\s+les\s+heures\b",
            r"\bà\s+partir\s+de\b", r"\bimminent", r"\bsous\s+peu\b",
            r"\bdernière\s+heure\b", r"\bbreaking\b", r"\bflash\b",
            r"\bavant\s+qu'il\b", r"\btrop\s+tard\b",
        ],
    },
}


class TemporalSensitivityScorer:
    def __init__(self):
        self.compiled = {}
        for category, config in TEMPORAL_PATTERNS.items():
            self.compiled[category] = {
                "weight": config["weight"],
                "patterns": [re.compile(p, re.IGNORECASE) for p in config["patterns"]],
            }

    def score(self, text):
        """
        Score temporal sensitivity [0, 1].
        0.0 = normal period, 1.0 = peak crisis/sensitive timing.
        """
        total = 0.0
        max_possible = 0.0

        for category, config in self.compiled.items():
            weight = config["weight"]
            matched = sum(1 for p in config["patterns"] if p.search(text))
            total_patterns = len(config["patterns"])

            if total_patterns > 0:
                cat_score = (matched / total_patterns) * weight
                total += cat_score
                max_possible += weight

        if max_possible == 0:
            return 0.0

        raw = total / max_possible
        boosted = min(1.0, raw * 4.0)
        return round(boosted, 4)
