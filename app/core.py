# app/core.py
import logging
import os
from datetime import datetime

# — logger (samme opsætning som før)
logger = logging.getLogger("rag-app")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# — IMPORTER ELLER DEFINER DINE HJÆLPEFUNKTIONER HER —
# Hvis de pt. ligger i app/__init__.py, flyt dem hertil 1:1:
#   md_to_plain, get_rag_answer, detect_language, detect_role,
#   role_label, detect_intent
#
# Eksempler (PLADSHOLDERE – brug dine rigtige funktioner):
def md_to_plain(md: str) -> str:
    # TODO: brug din eksisterende implementation
    return md

async def get_rag_answer(prompt: str) -> str:
    # TODO: brug din eksisterende implementation
    return prompt

def detect_language(text: str) -> str:
    # TODO
    return "da"

def detect_role(text: str, lang: str) -> str:
    # TODO
    return "clinician"

def role_label(lang: str, role: str) -> str:
    # TODO
    return "kliniker"

def detect_intent(text: str, lang: str) -> str:
    # TODO
    return "general"
