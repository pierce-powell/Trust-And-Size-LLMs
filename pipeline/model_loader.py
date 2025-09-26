# pipeline/model_loader.py
from typing import Tuple, Dict, Any, List
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    logging as hf_logging
)
hf_logging.set_verbosity_error()

class ModelLoader:
    """
    Two-step loader:
      1) prepare_metadata(models_config) -> creates mapping of keys -> meta (no weights loaded)
      2) load_instance(key, device, torch_dtype=None) -> loads weights & tokenizer, caches instance
      3) unload_instance(key) -> deletes weights and clears cache (and empties GPU cache)
    """

    def __init__(self):
        # metadata: key -> {model_id, alias, size_name, task_type}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        # instances cache: key -> {"model":..., "tokenizer":..., "task_type":...}
        self._instances: Dict[str, Dict[str, Any]] = {}

    def _make_key(self, alias: str, size_name: str) -> str:
        return f"{alias}::{size_name}"

    def prepare_metadata(self, models_config: List[Dict[str, Any]]):
        """
        Populate self.metadata from the models_config (YAML structure).
        Does NOT download model weights.
        """
        for entry in models_config:
            family = entry.get("family")
            alias = entry.get("alias", family)
            task_type = entry.get("task_type", "causal")
            sizes = entry.get("sizes", [])
            for s in sizes:
                model_id = s["id"]
                size_name = s.get("size_name", model_id)
                key = self._make_key(alias, size_name)
                self.metadata[key] = {
                    "model_id": model_id,
                    "alias": alias,
                    "size_name": size_name,
                    "task_type": task_type
                }
        return self.metadata

    def list_keys(self) -> List[str]:
        return list(self.metadata.keys())

    def is_loaded(self, key: str) -> bool:
        return key in self._instances

    def load_instance(self, key: str, device: torch.device = None, torch_dtype=None, use_auth_token: Any = None) -> Dict[str, Any]:
        """
        Load model + tokenizer for key and move model to `device`.
        - device: torch.device (if None, will use cuda if available else cpu)
        - torch_dtype: e.g. torch.float16, or None
        - use_auth_token: token or True to use HF token from env; passed to from_pretrained
        Returns dict {"model", "tokenizer", "task_type"}.
        Caches the instance in self._instances[key].
        """
        if key not in self.metadata:
            raise KeyError(f"Unknown model key: {key}")

        if key in self._instances:
            return self._instances[key]

        meta = self.metadata[key]
        model_id = meta["model_id"]
        task_type = meta["task_type"]

        # set device default
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # auth token support: prefer explicit argument, else use HF_TOKEN env var if present
        auth = use_auth_token
        if auth is None:
            auth = os.environ.get("HF_TOKEN", None)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, use_auth_token=auth)
        tokenizer.padding_side = getattr(tokenizer, "padding_side", "left")

        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        # Avoid loading in 8-bit/accelerate here; keep simple. Users can customize outside.

        if task_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs, use_auth_token=auth)
        elif task_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kwargs, use_auth_token=auth)
        elif task_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_id, **load_kwargs, use_auth_token=auth)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        # Move model to device
        model.to(device)
        model.eval()

        inst = {"model": model, "tokenizer": tokenizer, "task_type": task_type, "device": device}
        self._instances[key] = inst
        return inst

    def unload_instance(self, key: str):
        """
        Unload and free resources associated with the model instance.
        """
        if key not in self._instances:
            return
        inst = self._instances.pop(key)
        try:
            model = inst.get("model")
            # delete and free
            del model
        except Exception:
            pass
        # try to free GPU memory
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload_all(self):
        for k in list(self._instances.keys()):
            self.unload_instance(k)
        self._instances = {}
