"""
Audience scope estimator: how wide is the affected audience?
Detects national vs regional vs local scope based on entities and language.
"""

import re


NATIONAL_MARKERS = [
    r"\bmaroc\b", r"\bmarocain[es]?\b", r"\bnational[e]?\b", r"\btout\s+le\s+pays\b",
    r"\broyaume\b", r"\bensemble\s+du\s+territoire\b", r"\btout(?:es)?\s+les\s+régions?\b",
    r"\btout(?:es)?\s+les\s+villes?\b", r"\bpays\b", r"\bgouvernement\b",
    r"\bministère\b", r"\bparlement\b", r"\bconstitution", r"\broi\b",
    r"\bmohammed\s+vi\b", r"\bsouverain\b", r"\bpalais\s+royal\b",
    r"\bonu\b", r"\bnations\s+unies\b", r"\bconseil\s+de\s+sécurité\b",
    r"\btous\s+les\s+marocains?\b", r"\bpopulation\b",
]

REGIONAL_MARKERS = [
    r"\brégion\b", r"\brégional[e]?\b", r"\bprovince\b", r"\bpréfecture\b",
    r"\bwilaya\b", r"\bwali\b", r"\bgouverneur\b",
    r"\bcasablanca\b", r"\brabat\b", r"\bmarrakech\b", r"\bfès\b", r"\btanger\b",
    r"\bagadir\b", r"\bmeknès\b", r"\boujda\b", r"\bkénitra\b", r"\btétouan\b",
    r"\blaâyoune\b", r"\blaayoune\b", r"\bdakhla\b", r"\bnador\b", r"\bsafi\b",
    r"\bel\s+jadida\b", r"\bbéni\s+mellal\b", r"\bkhénifra\b",
    r"\bsouss\b", r"\borient", r"\bdraa\b", r"\btafilalet\b",
    r"\bal\s+hoceima\b", r"\bhoceima\b",
]

LOCAL_MARKERS = [
    r"\bcommune\b", r"\bquartier\b", r"\bdouar\b", r"\bvillage\b",
    r"\bhabitan(?:t|ts)\b", r"\bvoisin", r"\blocal[e]?\b",
    r"\bcommunal[e]?\b", r"\bzone\s+rurale\b",
]


class AudienceScopeScorer:
    def __init__(self):
        self.national = [re.compile(p, re.IGNORECASE) for p in NATIONAL_MARKERS]
        self.regional = [re.compile(p, re.IGNORECASE) for p in REGIONAL_MARKERS]
        self.local = [re.compile(p, re.IGNORECASE) for p in LOCAL_MARKERS]

    def score(self, text):
        """
        Score the audience scope [0, 1].
        0.0 = very local, 0.5 = regional, 1.0 = national/international.
        """
        nat_hits = sum(1 for p in self.national if p.search(text))
        reg_hits = sum(1 for p in self.regional if p.search(text))
        loc_hits = sum(1 for p in self.local if p.search(text))

        # National dominates, then regional, then local
        if nat_hits >= 2:
            base = 0.85
        elif reg_hits >= 1:
            base = 0.50
        elif loc_hits >= 1:
            base = 0.20
        else:
            base = 0.30  # Default: moderate scope

        return round(base, 4)
