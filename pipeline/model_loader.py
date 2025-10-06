# pipeline/model_loader.py
import os
import sys
import importlib.util
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

    if not os.path.isdir(cache_root):
        return None

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


def _import_py_files_from_path(path: str):
    """
    Try to import python files from the path so custom classes (tokenizers, modeling code) get registered.
    This executes code in those modules — only use with snapshots you trust.
    """
    if not os.path.isdir(path):
        return
    # import top-level .py files and walk subdirectories
    for root, _, files in os.walk(path):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(root, fn)
            # create a deterministic module name from file path
            rel = os.path.relpath(full, path).replace(os.sep, ".")
            mod_name = f"model_snapshot_{rel[:-3]}"  # strip .py
            try:
                if mod_name in sys.modules:
                    continue
                spec = importlib.util.spec_from_file_location(mod_name, full)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    sys.modules[mod_name] = mod
            except Exception as e:
                # don't crash import attempts — print for debugging and continue
                print(f"Warning: failed to import {full}: {e}")


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
        This will:
          - find the local snapshot directory,
          - add the snapshot (and snapshot/src) to sys.path,
          - attempt to import .py files from the snapshot to register custom classes,
          - call AutoTokenizer/AutoModelForCausalLM with trust_remote_code=True and local_files_only=True.
        """
        meta = self.models_meta.get(key)
        if meta is None:
            raise ValueError(f"Unknown model key: {key}")

        model_id = meta["model_id"]
        local_path = find_local_model_path(model_id)
        if not local_path:
            raise FileNotFoundError(f"Could not find local model path for {model_id} under TRANSFORMERS_CACHE.")

        print(f"Loading {key} from {local_path}")

        # Ensure python can import files from the snapshot dir (and src/ if present)
        if local_path not in sys.path:
            sys.path.insert(0, local_path)
        src_dir = os.path.join(local_path, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Try importing python files inside snapshot (this executes them) so custom tokenizers are registered.
        # WARNING: executes arbitrary code from the snapshot; only do this with trusted snapshots.
        _import_py_files_from_path(local_path)
        if os.path.isdir(src_dir):
            _import_py_files_from_path(src_dir)

        # Try to load tokenizer & model; use trust_remote_code=True so tokenizers implemented in the snapshot get used.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                use_fast=False,
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            # helpful diagnostic
            print(f"AutoTokenizer.from_pretrained failed for {local_path}: {e}")
            # re-raise so caller can see logs; you could implement additional heuristics here
            raise

        try:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"AutoModelForCausalLM.from_pretrained failed for {local_path}: {e}")
            raise

        return {"model": model, "tokenizer": tokenizer, "device": device, "alias": meta["alias"]}

    def unload_instance(self, key: str):
        """Unload model from memory (no-op for CPU; could add torch.cuda.empty_cache() if using GPU)."""
        pass
