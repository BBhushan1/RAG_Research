import os
import logging
from dotenv import load_dotenv
from datetime import datetime


if not load_dotenv():
    logging.warning("No .env file found — using defaults. Ensure this is intentional.")

# Enhanced Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("rag_pipeline.log", encoding="utf-8", mode="a"),
        logging.StreamHandler()  
    ]
)

# LLM & Ollama
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

if not isinstance(LLM_MODEL, str) or not LLM_MODEL.strip():
    raise ValueError("LLM_MODEL must be a non-empty string")
if not OLLAMA_URL.startswith(("http://", "https://")):
    raise ValueError("OLLAMA_URL must be a valid HTTP/HTTPS URL")

try:
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", 15))
    START_YEAR = int(os.getenv("START_YEAR", 2020))
    END_YEAR = int(os.getenv("END_YEAR", datetime.now().year))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.75))
    
    if MAX_RESULTS <= 0 or MAX_RESULTS > 50:
        raise ValueError("MAX_RESULTS must be between 1 and 50")
    if START_YEAR > END_YEAR:
        raise ValueError("START_YEAR cannot be after END_YEAR")
    if not 0 <= SIMILARITY_THRESHOLD <= 1:
        raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
        
except ValueError as e:
    logging.error(f"Invalid configuration: {str(e)}")
    raise

OUTPUT_DIR = os.path.abspath(os.getenv("OUTPUT_DIR", "./output"))
VECTOR_DB_PATH = os.path.abspath(os.getenv("VECTOR_DB_PATH", "./vector_db"))
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "./data"))
CACHE_DIR = os.path.abspath(os.getenv("CACHE_DIR", "./cache"))

LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", 16384))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 8192))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))

for path in [OUTPUT_DIR, VECTOR_DB_PATH, DATA_DIR, CACHE_DIR]:
    try:
        os.makedirs(path, exist_ok=True)
        # Test write permission
        test_file = os.path.join(path, ".write_test")
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logging.info(f"Directory ensured and writable: {path}")
    except Exception as e:
        logging.error(f"Failed to create or write to directory {path}: {str(e)}")
        raise

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.info(f"Pipeline initialized. Run ID: {RUN_ID}")
logging.info(f"Config: LLM={LLM_MODEL}, Max Results={MAX_RESULTS}, Years={START_YEAR}-{END_YEAR}")