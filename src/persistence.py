import pickle
import numpy as np
from pathlib import Path

def save_chunks(chunks: list, path: Path):
    with open(path, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_embeddings(embeddings: np.ndarray, path: Path):
    np.save(path, embeddings)

def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)