"""
guardrails.py — Pre-retrieval safety checks (runs BEFORE retrieval and BEFORE the LLM)

Implements the documented requirements:
  1. Emergency detection  -> predefined keyword list -> "Call 999"
  2. Prompt injection     -> filtered
  3. Scope limitation     -> non-oral-health queries blocked with:
                             "Ask about oral health question only"
Blocked queries never reach the LLM.

Emergency detection is deliberately a plain keyword match (as specified):
any query containing a keyword from the list triggers the "Call 999" message.
"""

import re
import unicodedata


class Guardrails:
    def __init__(self):
        # ---------- Emergency keywords (predefined list, as in FYP1) ----------
        self.emergency_keywords = [
            "chest pain",
            "heart attack",
            "seizure",
            "unconscious",
            "difficulty breathing",
            "severe bleeding",
            "stroke",
            "swollen face",
            "dental abscess",
            "uncontrolled bleeding after extraction",
            "broken jaw",
        ]

        # ---------- Prompt injection ----------
        self.injection_patterns = [
            "ignore previous",
            "override",
            "system prompt",
            "you are now",
            "your new role",
            "forget your instructions",
            "system:",
        ]
        # ---------- Prompt injection: extra patterns (catch rewordings) ----------
        self.injection_regexes = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in [
                r"\bignore\b.{0,30}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}\b(instructions?|prompts?|rules?|directions?)\b",
                r"\bdisregard\b.{0,30}\b(instructions?|prompts?|rules?)\b",
                r"\bforget\b.{0,30}\b(instructions?|prompts?|rules?)\b",
                r"\b(reveal|show|print|repeat|display|leak)\b.{0,25}\b(system|hidden|initial|secret)\b.{0,10}\b(prompt|instructions?)\b",
                r"\byour new (role|instructions?|task|rules?)\b",
                r"\bpretend (to be|you are|you're)\b",
                r"\b(jailbreak|developer mode|do anything now)\b",
                r"(^|\n)\s*(system|assistant)\s*:",
                r"<\|?(im_start|im_end|system)\|?>",
            ]
        ]

        # ---------- Scope keywords (oral health domain) ----------
        self.scope_keywords = [
            # General oral health
            "oral", "mouth", "tongue", "lip", "cheek", "palate", "saliva", "jaw",
            # Dental anatomy
            "tooth", "teeth", "enamel", "dentin", "pulp", "root", "crown",
            "molar", "incisor", "fissure",
            # Gums
            "gum", "gingiva", "gingival", "periodontal", "periodontitis",
            "gingivitis", "peri-implantitis", "implantitis",
            # Procedures & prevention
            "brush", "brushing", "toothbrush", "toothpaste", "floss", "flossing",
            "mouthwash", "fluoride", "fluoridated", "varnish", "sealant",
            "resin", "restoration", "restorative", "prr", "filling", "extraction",
            "root canal", "implant", "denture", "braces", "orthodontic",
            "orthodontist", "malocclusion",
            # Conditions
            "caries", "cavity", "decay", "plaque", "tartar", "abscess",
            "infection", "ulcer", "lesion", "leukoplakia", "erythroplakia",
            "malignant", "cancer", "sugar", "sugary",
            # Programmes / documents
            "icdas", "cppc", "cpg", "moh", "tobacco", "smoking", "smoke",
            "vape", "betel", "quid",
            # Population groups
            "pregnant", "pregnancy", "mother", "child", "children", "pediatric",
            "paediatric", "preschool", "kindergarten", "teacher", "elderly",
            "older", "adult", "baby", "infant", "toddler",
            # Dental visits & professionals
            "dentist", "dental", "hygienist", "checkup", "cleaning", "xray",
            "radiograph",
            # Common symptoms
            "toothache", "bleeding gums", "bad breath", "halitosis",
            "sensitive teeth", "dry mouth", "xerostomia",
            # Treatments & materials
            "amalgam", "composite", "ceramic", "bridge", "veneer", "whitening",
            # Related medical conditions
            "diabetes",
            # Clinical terms used in the MOH guidelines
            "hall technique", "silver diamine", "sdf", "pulpotomy", "pulpectomy",
            "primary molar", "primary teeth", "deciduous", "preformed crown",
            "debridement", "scaling", "periodontist", "mucosa", "mucosal",
            "submucous", "bitewing", "occlusal", "proximal", "mouthguard",
            "mouth guard", "tooth wear", "erosion", "fluorosis", "antenatal",
            "edentulous", "geriatric", "ageing", "aging", "bottle feeding",
            "breastfeeding", "lift the lip", "anticipatory guidance",
            "toothbrushing",
        ]
        # Whole-word match (so "gum" does not match "argument"); plural allowed
        self.scope_regexes = [
            re.compile(r"\b" + re.escape(kw) + r"(s|es)?\b")
            for kw in self.scope_keywords
        ]

    @staticmethod
    def _normalise(text):
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = re.sub(r"[ \t]+", " ", text)
        return text.lower().strip()

    def process_query(self, query, mode=None):
        """
        `mode` is accepted so callers can pass it, but the checks are the same
        for both modes (as specified).

        Returns dict:
          {
            "blocked": bool,
            "message": str,     # shown to user if blocked, else "OK"
            "reason": str,      # "emergency" | "injection" | "out_of_scope" | "ok"
          }
        """
        q = self._normalise(query)

        # 1. Emergency detection: predefined keyword list -> "Call 999"
        for kw in self.emergency_keywords:
            if kw in q:
                return {
                    "blocked": True,
                    "message": "🚨 Emergency detected. Call 999 immediately.",
                    "reason": "emergency",
                }

        # 2. Prompt injection filtering
        if any(pat in q for pat in self.injection_patterns) or any(
            rx.search(q) for rx in self.injection_regexes
        ):
            return {
                "blocked": True,
                "message": (
                    "⚠️ Prompt injection attempt blocked. "
                    "Your request cannot be processed due to policy restrictions."
                ),
                "reason": "injection",
            }

        # 3. Scope limitation: block non-oral-health queries
        if not any(rx.search(q) for rx in self.scope_regexes):
            return {
                "blocked": True,
                "message": "Ask about oral health question only",
                "reason": "out_of_scope",
            }

        # All checks passed
        return {"blocked": False, "message": "OK", "reason": "ok"}


# ---------- Singleton ----------
_guardrails = None


def get_guardrails():
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails()
    return _guardrails