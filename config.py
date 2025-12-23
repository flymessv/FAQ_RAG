import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    kb_dir: str = os.getenv("KB_DIR", "kb")
    index_dir: str = os.getenv("INDEX_DIR", "data/index")

    top_k: int = _get_int("TOP_K", 3)
    min_sim: float = _get_float("MIN_SIM", 0.25)
    max_context_chars: int = _get_int("MAX_CONTEXT_CHARS", 6000)
    max_output_tokens: int = _get_int("MAX_OUTPUT_TOKENS", 300)

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

def get_logger(name: str = "faq_bot") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logger.setLevel(level)
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger
