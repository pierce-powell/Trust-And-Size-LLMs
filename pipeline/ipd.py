# pipeline/ipd.py
from typing import List, Tuple, Dict, Any, Callable, Optional
import random
import torch
from pipeline.utils import get_device
import re

# Payoff matrix for Prisoner's Dilemma (R,T,S,P) common form
# We'll return a tuple (score_agent1, score_agent2)
PAYOFF = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1)
}

def always_cooperate(history, **kwargs):
    return "C"

def always_defect(history, **kwargs):
    return "D"

def tit_for_tat(history, **kwargs):
    # cooperate first, then mirror opponent last move
    if not history:
        return "C"
    # history is list of (a1,a2) tuples
    last = history[-1]
    if kwargs.get("player_index", 0) == 0:
        opp_last = last[1]
    else:
        opp_last = last[0]
    return opp_last

def two_tits_for_tat(history, **kwargs):
    """
    Two-Tits-For-Tat (2TFT / 2T4T alias)
    Cooperate unless the opponent has defected in each of the last TWO rounds.
    If the opponent defected in both last rounds, retaliate (defect); otherwise cooperate.
    """
    if len(history) < 2:
        return "C"
    # get opponent last two moves
    if kwargs.get("player_index", 0) == 0:
        opp_moves = [history[-1][1], history[-2][1]]
    else:
        opp_moves = [history[-1][0], history[-2][0]]
    # if opponent defected in both last two rounds -> defect
    if opp_moves[0] == "D" and opp_moves[1] == "D":
        return "D"
    return "C"

# map names -> functions (include alias '2T4T' to accept user's shorthand)
BASE_STRATEGIES = {
    "ALLC": always_cooperate,
    "ALLD": always_defect,
    "TFT": tit_for_tat,
    "2TFT": two_tits_for_tat,
    "2T4T": two_tits_for_tat,
    "RANDOM": random_strategy
}


class LLMAgent:
    """
    Wraps a HF model+tokenizer and a set of prompt templates (3 prompts).
    The prompt used per run is chosen by prompt_index (0..2).
    The agent produces actions "C" or "D".
    """
    def __init__(self, model, tokenizer, prompt_templates: List[str], device=None, max_new_tokens=32):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_templates = prompt_templates
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.max_new_tokens = max_new_tokens

    def _sanitize_output(self, text: str) -> str:
        # try to extract the single-letter choice or word
        text = text.strip().upper()
        # prefer explicit letters
        m = re.search(r'\b([CD])\b', text)
        if m:
            return m.group(1)
        if "COOPERATE" in text:
            return "C"
        if "DEFECT" in text:
            return "D"
        # fallback to first char if C or D
        if len(text) > 0 and text[0] in ("C","D"):
            return text[0]
        # fallback random
        return random.choice(["C","D"])

    def choose(self, history: List[Tuple[str,str]], prompt_index: int, player_index: int = 0) -> str:
        """
        history: list of tuples (a1,a2) with previous moves
        prompt_index: index into prompt_templates
        player_index: 0 or 1 indicating whether this agent is player1 or player2
        """
        tmpl = self.prompt_templates[prompt_index]
        # create a short readable history string
        hist_lines = []
        for i,(a1,a2) in enumerate(history):
            hist_lines.append(f"Round {i+1}: Player1={a1}, Player2={a2}")
        history_str = "\n".join(hist_lines) if hist_lines else "No rounds yet."

        prompt = tmpl.format(history=history_str, you="Player1" if player_index==0 else "Player2")
        # append explicit instruction
        prompt += "\n\nChoose one action only: 'C' for Cooperate or 'D' for Defect. Reply with the single letter."

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # often returns prompt+answer; try to extract the answer
        # we'll try to remove the prompt prefix
        if text.startswith(prompt):
            answer_text = text[len(prompt):].strip()
        else:
            # search for last line after prompt-like punctuation
            answer_text = text.strip()
        return self._sanitize_output(answer_text)

class IPDGame:
    """
    Orchestrates an iterated PD match between two players, where either can be:
      - a function(history, **kwargs) -> "C"/"D" (base strategy), OR
      - an LLMAgent instance
    """
    def __init__(self, player1, player2, rounds:int=50, seed:Optional[int]=None):
        self.p1 = player1
        self.p2 = player2
        self.rounds = rounds
        if seed is not None:
            random.seed(seed)

    def _step_choice(self, player, history, player_index):
        if isinstance(player, LLMAgent):
            return player.choose(history, prompt_index=player.prompt_choice if hasattr(player, "prompt_choice") else 0, player_index=player_index)
        elif callable(player):
            # base strategy
            return player(history=history, player_index=player_index)
        else:
            raise ValueError("Unknown player type")

    def play(self, p1_prompt_index:int=0, p2_prompt_index:int=0) -> Dict[str, Any]:
        """
        If players are LLMAgents, set attribute prompt_choice for the duration.
        Returns dict:
          "history": [(a1,a2),...], "scores": (sum1,sum2), "cooperation_count": (c1,c2)
        """
        # set temporary prompt choices if agents
        if isinstance(self.p1, LLMAgent):
            setattr(self.p1, "prompt_choice", p1_prompt_index)
        if isinstance(self.p2, LLMAgent):
            setattr(self.p2, "prompt_choice", p2_prompt_index)

        history: List[Tuple[str,str]] = []
        total1 = total2 = 0
        coop1 = coop2 = 0

        for _ in range(self.rounds):
            a1 = self._step_choice(self.p1, history, player_index=0)
            a2 = self._step_choice(self.p2, history, player_index=1)
            # normalize
            a1 = a1 if a1 in ("C","D") else "D"
            a2 = a2 if a2 in ("C","D") else "D"
            history.append((a1,a2))
            s1,s2 = PAYOFF[(a1,a2)]
            total1 += s1
            total2 += s2
            coop1 += (1 if a1=="C" else 0)
            coop2 += (1 if a2=="C" else 0)

        # cleanup prompt_choice attributes
        if isinstance(self.p1, LLMAgent):
            delattr(self.p1, "prompt_choice")
        if isinstance(self.p2, LLMAgent):
            delattr(self.p2, "prompt_choice")

        return {
            "history": history,
            "scores": (total1, total2),
            "cooperations": (coop1, coop2),
            "rounds": self.rounds
        }
