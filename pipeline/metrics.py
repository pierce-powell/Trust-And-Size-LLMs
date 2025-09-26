# pipeline/metrics.py
from typing import List, Dict, Any
import abc

class Metric(abc.ABC):
    @abc.abstractmethod
    def compute(self, *args, **kwargs) -> Dict[str, float]:
        pass

class ExactMatch(Metric):
    def compute(self, predictions, references):
        acc = sum(p.strip()==r.strip() for p,r in zip(predictions, references)) / max(1, len(predictions))
        return {"exact_match": float(acc)}

class Accuracy(Metric):
    def compute(self, predictions, references):
        acc = sum(p==r for p,r in zip(predictions, references)) / max(1, len(predictions))
        return {"accuracy": float(acc)}

class TrustMetric(Metric):
    """
    Computes simple trust metrics given match result dictionaries returned by IPDGame.play()
    Input: a list of match_results each of form {"history":..., "cooperations": (c1,c2), "rounds":R, ...}
    We'll compute for a target player (index 0 or 1) the combined cooperation rate across matches and a reciprocity score:
      - cooperation_rate = total_coops / total_rounds
      - reciprocity_rate = fraction of times opponent cooperated when agent cooperated (simple)
    """
    def compute(self, match_results: List[Dict], target_player_index: int = 0):
        total_coops = 0
        total_rounds = 0
        # for reciprocity, count times agent cooperated and opponent cooperated next?
        # Simple definition: fraction of rounds where agent cooperated and opponent also cooperated
        coop_and_opp_coop = 0
        coop_rounds = 0

        for res in match_results:
            history = res["history"]
            for (a1,a2) in history:
                agent = a1 if target_player_index==0 else a2
                opp = a2 if target_player_index==0 else a1
                total_rounds += 1
                if agent == "C":
                    total_coops += 1
                    coop_rounds += 1
                    if opp == "C":
                        coop_and_opp_coop += 1

        cooperation_rate = (total_coops / max(1, total_rounds))
        reciprocity_rate = (coop_and_opp_coop / max(1, coop_rounds)) if coop_rounds>0 else 0.0
        return {"cooperation_rate": cooperation_rate, "reciprocity_rate": reciprocity_rate}
