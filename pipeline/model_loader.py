import os
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

def find_local_model_path(model_id: str) -> Optional[str]:
    """
    Locate local snapshot path for a model_id under TRANSFORMERS_CACHE.
    Example: models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<hash>/
    """
    cache_root = os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))
    owner_repo = model_id.replace("/", "--")
    pattern = f"models--{owner_repo}"

    for entry in os.listdir(cache_root):
        if entry.startswith(pattern):
            model_root = os.path.join(cache_root, entry)
            snapshots_dir = os.path.join(model_root, "snapshots")
            if os.path.isdir(snapshots_dir):
                for snap in os.listdir(snapshots_dir):
                    snap_path = os.path.join(snapshots_dir, snap)
                    if os.path.exists(os.path.join(snap_path, "config.json")):
                        return snap_path
    return None


class ModelLoader:
    def __init__(self):
        self.models_meta = {}

    def prepare_metadata(self, models_cfg: List[Dict[str, Any]]):
        """
        Parse the YAML 'models:' section into internal metadata.
        """
        for m in models_cfg:
            alias = m.get("alias", m["family"])
            for s in m.get("sizes", []):
                size = s.get("size_name")
                model_id = s.get("id")
                key = f"{alias}::{size}"
                self.models_meta[key] = {
                    "alias": alias,
                    "size": size,
                    "model_id": model_id,
                    "task_type": m.get("task_type", "causal")
                }
        return self.models_meta

    def list_keys(self) -> List[str]:
        """Return all keys in deterministic order."""
        return list(self.models_meta.keys())

    def load_instance(self, key: str, device: str = "cpu", use_auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Given a key like 'qwen2.5::0.5B', load the model/tokenizer offline if possible.
        """
        meta = self.models_meta.get(key)
        if meta is None:
            raise ValueError(f"Unknown model key: {key}")

        model_id = meta["model_id"]
        local_path = find_local_model_path(model_id)
        if not local_path:
            raise FileNotFoundError(f"Could not find local model path for {model_id} under TRANSFORMERS_CACHE.")

        print(f"Loading {key} from {local_path}")
        tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True)
        return {"model": model, "tokenizer": tokenizer, "device": device, "alias": meta["alias"]}

    def unload_instance(self, key: str):
        """Unload model from memory (no-op for CPU; could add torch.cuda.empty_cache() if using GPU)."""
        pass
