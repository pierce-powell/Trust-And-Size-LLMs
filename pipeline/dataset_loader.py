from typing import List, Dict, Any
import json

def load_simple_json(path: str) -> List[Dict[str, Any]]:
    """
    Expect a JSON lines file or JSON list of dicts with fields:
      - input
      - target
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        if text.startswith('['):
            data = json.loads(text)
        else:
            # assume jsonlines
            for line in text.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data
