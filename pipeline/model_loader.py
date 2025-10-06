# pipeline/model_loader.py
import os
import logging
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

def _get_cache_roots():
    """
    Return a list of plausible transformer cache roots to search.
    Order: TRANSFORMERS_CACHE, HF_HOME/transformers, default ~/.cache/huggingface/transformers
    """
    roots = []
    tcache = os.environ.get("TRANSFORMERS_CACHE")
    if tcache:
        roots.append(tcache)
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HF_HOME".upper())
    if hf_home:
        roots.append(os.path.join(hf_home, "transformers"))
    # default cache location used by huggingface
    default = os.path.expanduser("~/.cache/huggingface/transformers")
    roots.append(default)
    return roots

def find_local_model_path(model_id: str) -> Optional[str]:
    """
    Locate a local model directory inside TRANSFORMERS_CACHE, given a Hugging Face model ID.
    Example:
      model_id = "Qwen/Qwen2.5-0.5B-Instruct"
      TRANSFORMERS_CACHE = /home/pi047867/hf_cache/hf_cache
      → looks for models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<hash>/
    """
    cache_root = os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))
    owner_repo = model_id.replace("/", "--")
    pattern = f"models--{owner_repo}"

    for entry in os.listdir(cache_root):
        if entry.startswith(pattern):
            model_root = os.path.join(cache_root, entry)
            snapshots_dir = os.path.join(model_root, "snapshots")
            if os.path.isdir(snapshots_dir):
                # pick first snapshot (usually only one)
                for snap in os.listdir(snapshots_dir):
                    snap_path = os.path.join(snapshots_dir, snap)
                    if os.path.exists(os.path.join(snap_path, "config.json")):
                        return snap_path
    return None



class ModelLoader:
    """
    Lightweight loader that supports local-only loading when pre-downloaded cache exists.
    """
    def __init__(self):
        # you can preload metadata here if needed
        pass

    def prepare_metadata(self, models_cfg):
        # Example: convert your yaml models block to a flat list of model ids / aliases
        # This implementation assumes models_cfg is the YAML structure you have (with sizes[] containing id)
        metadata = []
        for m in models_cfg:
            alias = m.get("alias") or m.get("family")
            sizes = m.get("sizes", [])
            for s in sizes:
                mid = s.get("id")
                if mid:
                    metadata.append({"alias": alias, "model_id": mid, "size_name": s.get("size_name")})
        return metadata

    def list_keys(self):
        # if you use prepare_metadata above, just return alias::size pairs or model ids
        # but main expects some `keys`, so return a list of model_id strings for simplicity
        # You may adapt this to your existing code's key format
        raise NotImplementedError("list_keys must be implemented according to your pipeline metadata format.")

    def load_instance(self, model_id: str, device: str = "cpu", use_auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Load tokenizer and model. If a local path is found, load from there using local_files_only=True.
        Returns a dict with keys: model, tokenizer, device, alias (optional)
        """
        logger.info("Loading model: %s (device=%s)", model_id, device)
        local_path = find_local_model_path(model_id)
        local_files_only = True
        hf_ref = local_path if local_path is not None else model_id

        # Force use_fast=False for robustness (fast tokenizers sometimes need conversions)
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_ref, use_fast=False, local_files_only=local_files_only)
        except Exception as e:
            logger.warning("AutoTokenizer.from_pretrained failed for %s with local_files_only=%s: %s", hf_ref, local_files_only, e)
            # retry using use_fast=True as fallback (or raise if you prefer)
            tokenizer = AutoTokenizer.from_pretrained(hf_ref, use_fast=True, local_files_only=local_files_only)

        try:
            model = AutoModelForCausalLM.from_pretrained(hf_ref, local_files_only=local_files_only)
        except Exception as e:
            logger.error("AutoModelForCausalLM.from_pretrained failed for %s (local=%s): %s", hf_ref, local_path is not None, e)
            raise

        return {"model": model, "tokenizer": tokenizer, "device": device, "alias": model_id}
