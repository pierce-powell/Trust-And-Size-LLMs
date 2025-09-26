import json
from typing import Dict, Any

class PromptManager:
    """
    Loads prompt templates from a JSON file or dict.
    Prompts are simple Python .format templates, e.g.:
      "prompt1": "Translate to French: {input}"
    """

    def __init__(self, prompt_file: str = None, prompts: Dict[str,str] = None):
        if prompt_file:
            with open(prompt_file, 'r') as f:
                self.prompts = json.load(f)
        else:
            self.prompts = prompts or {}

    def get(self, name: str, **kwargs) -> str:
        template = self.prompts.get(name)
        if template is None:
            raise KeyError(f"Prompt {name} not found")
        return template.format(**kwargs)

    def register(self, name: str, template: str):
        self.prompts[name] = template
