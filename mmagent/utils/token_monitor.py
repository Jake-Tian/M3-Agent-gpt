# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Token consumption monitor for memory and control phases.

import json
import os
import threading
from pathlib import Path


class TokenMonitor:
    """Thread-safe token consumption monitor. Tracks tokens for memory and control phases."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, save_path=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, save_path=None):
        if self._initialized:
            return
        self._initialized = True
        self.save_path = save_path or "data/results/token_consumption.json"
        self._data = {
            "memory": {"total": 0, "generation": 0, "embedding": 0, "by_video": {}},
            "control": {"total": 0, "by_video": {}},
        }
        self._data_lock = threading.Lock()

    def add(self, phase, amount, subkey=None, video_id=None):
        """Add tokens. phase: 'memory' or 'control'. subkey: 'generation' or 'embedding' for memory."""
        with self._data_lock:
            if phase not in self._data:
                self._data[phase] = {"total": 0, "generation": 0, "embedding": 0, "by_video": {}}
            self._data[phase]["total"] += amount
            if subkey:
                if subkey not in self._data[phase]:
                    self._data[phase][subkey] = 0
                self._data[phase][subkey] += amount
            if video_id:
                if "by_video" not in self._data[phase]:
                    self._data[phase]["by_video"] = {}
                if video_id not in self._data[phase]["by_video"]:
                    self._data[phase]["by_video"][video_id] = 0
                self._data[phase]["by_video"][video_id] += amount

    def save(self, path=None):
        path = path or self.save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._data_lock:
            with open(path, "w") as f:
                json.dump(self._data, f, indent=2)

    def load(self, path=None):
        """Load existing data and merge. Call at start of run to accumulate across processes."""
        path = path or self.save_path
        if os.path.exists(path):
            try:
                with open(path) as f:
                    loaded = json.load(f)
                with self._data_lock:
                    for phase in ["memory", "control"]:
                        if phase in loaded:
                            for k, v in loaded[phase].items():
                                if k == "by_video" and isinstance(v, dict):
                                    for vid, tok in v.items():
                                        self._data[phase]["by_video"][vid] = (
                                            self._data[phase]["by_video"].get(vid, 0) + tok
                                        )
                                elif isinstance(v, (int, float)):
                                    self._data[phase][k] = self._data[phase].get(k, 0) + v
            except (json.JSONDecodeError, IOError):
                pass

    def get_data(self):
        with self._data_lock:
            return json.loads(json.dumps(self._data))


# Global instance for easy import
_monitor = None


def get_monitor(save_path=None):
    global _monitor
    if _monitor is None:
        _monitor = TokenMonitor(save_path)
    return _monitor


def add_tokens(phase, amount, subkey=None, video_id=None):
    get_monitor().add(phase, amount, subkey, video_id)


def save_tokens(path=None):
    get_monitor().save(path)
