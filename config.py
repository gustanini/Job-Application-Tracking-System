# config.py
"""
This module centralizes all configurations.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "CSV_PATH": os.path.join(BASE_DIR, "data/jobs.csv"),
    "CV_PATH": os.path.join(BASE_DIR, "data/cv.txt"),
    "TOP_N_KEYWORDS": 10,
    "TOP_N_JOBS": 5,
    "LOG_LEVEL": "INFO",
    "CUSTOM_STOP_WORDS": {
        "experience", "required", "preferred", "responsibilities",
        "strong", "ability", "work", "skills",
        "team", "analytics", "requirements", "systems",
        "knowledge", "job", "years", "information",
        "quality", "company", "opportunity", "technical"
    },
}
