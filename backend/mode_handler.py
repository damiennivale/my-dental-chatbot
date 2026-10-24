"""
Mode-specific response generation with LLM
Uses Ollama local LLM for natural language responses
Falls back to template-based if LLM unavailable
"""

from typing import List, Dict
import re

class ModeHandler:
    def __init__(self):
        # LLM Configuration
        self.use_llm = False  # Will try to enable if ollama available
        self.model_name = "llama3.2:3b"
        
        # Try to initialize LLM
        try:
            import ollama
            self.ollama = ollama
            # Test if ollama is running
            self.ollama.list()
            self.use_llm = True
            print("✅ LLM (Ollama) initialized successfully")
        except ImportError:
            print("⚠️ Ollama not installed. Using template-based responses.")
        except Exception as e:
            print(f"⚠️ Ollama error: {e}. Using template-based responses.")
        
        # Medical terminology mapping for patient mode
        self.medical_to_plain = {
            "contraindications": "when NOT to use",
            "indications": "when to use",
            "hypertension": "high blood pressure",
            "hypoglycemia": "low blood sugar",
            "hyperglycemia": "high blood sugar",
            "eGFR": "kidney function test",
            "HbA1c": "blood sugar level over 3 months",
            "ACE inhibitors": "blood pressure medication",
            "ARBs": "blood pressure medication",
            "metformin": "diabetes medication",
            "insulin": "blood sugar controlling hormone",
            "dosage": "how much to take",
            "administration": "how to take it",
            "adverse effects": "side effects",
            "monitoring": "checking your health",
            "titrate": "slowly adjust",
            "gastrointestinal": "stomach and gut",
            "hepatic impairment": "liver problems",
            "renal function": "kidney health",
            "sepsis": "serious infection",
            "diabetic ketoacidosis": "dangerous high blood sugar emergency"
        }
    
    def generate_professional_response(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate clinical response using LLM (with fallback to template)"""
        
        if not retrieved_chunks:
            return self._no_results_response("professional")
        
        # Try LLM first if available
        if self.use_llm:
            try:
                return self._llm_professional_response(query, retrieved_chunks)
            except Exception as e:
                print(f"LLM error: {e}, falling back to template")
        
        # Fallback to template-based response
        return self._template_professional_response(query, retrieved_chunks)
    
    def generate_patient_response(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate patient-friendly response using LLM (with fallback to template)"""
        
        if not retrieved_chunks:
            return self._no_results_response("patient")
        
        # Try LLM first if available
        if self.use_llm:
            try:
                return self._llm_patient_response(query, retrieved_chunks)
            except Exception as e:
                print(f"LLM error: {e}, falling back to template")
        
        # Fallback to template-based response
        return self._template_patient_response(query, retrieved_chunks)
    
    def _llm_professional_response(self, query: str, chunks: List[Dict]) -> str:
        """Generate clinical response using local LLM"""
        
        # Prepare context from retrieved chunks
        context_parts = []
        citations = []
        
        for i, chunk in enumerate(chunks[:5], 1):
            context_parts.append(f"[Source {i}]\n{chunk['text']}")
            citations.append(f"{i}. {chunk['metadata'].get('title', 'CPG Document')} (Score: {chunk.get('similarity', 'N/A')})")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Professional mode prompt
        prompt = f"""You are a clinical decision support system for Malaysian Ministry of Health (MOH) Clinical Practice Guidelines.

**ROLE:** Medical Professional / Oral Health Personnel
**LANGUAGE:** Use appropriate medical terminology
**REQUIREMENTS:**
- Answer based ONLY on the provided CPG guidelines
- Include specific clinical values (eGFR, HbA1c, blood pressure, dosages)
- Structure your answer with clear sections
- Cite sources using [Source X] notation
- If information is not in the guidelines, say "Not found in the provided CPG documents"

**GUIDELINES:**
{context}

**QUESTION:** {query}

**ANSWER (Clinical Format):"""

        # Call local LLM
        response = self.ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}  # Low temperature for factual responses
        )
        
        answer = response['message']['content']
        
        # Add citations at the end
        answer += f"\n\n---\n**📖 References:**\n" + "\n".join(citations)
        
        return answer
    
    def _llm_patient_response(self, query: str, chunks: List[Dict]) -> str:
        """Generate patient-friendly response using local LLM"""
        
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(chunks[:5], 1):
            context_parts.append(f"[Source {i}]\n{chunk['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Patient mode prompt
        prompt = f"""You are a health education assistant for Malaysian Ministry of Health (MOH) Clinical Practice Guidelines.

**ROLE:** Patient Health Educator
**LANGUAGE:** Simple, clear, non-technical language (explain medical terms)
**REQUIREMENTS:**
- Answer based ONLY on the provided CPG guidelines
- Use everyday language that a patient would understand
- Focus on what the patient needs to DO
- Include warning signs to watch for
- Encourage talking to their healthcare provider
- Be reassuring but honest

**GUIDELINES:**
{context}

**QUESTION:** {query}

**ANSWER (Patient-Friendly Format):"""

        response = self.ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3}  # Slightly higher for natural language
        )
        
        answer = response['message']['content']
        
        # Add a friendly footer
        answer += "\n\n---\n*💡 Remember to discuss any health concerns with your healthcare provider.*"
        
        return answer
    
    def _template_professional_response(self, query: str, chunks: List[Dict]) -> str:
        """Template-based clinical response (fallback)"""
        
        # Build citations
        citations = []
        for i, chunk in enumerate(chunks[:3], 1):
            citations.append(f"{i}. {chunk['metadata'].get('title', 'CPG Document')} (Similarity: {chunk.get('similarity', 'N/A')})")
        
        # Extract key clinical information
        clinical_info = []
        for chunk in chunks[:3]:
            text = chunk['text']
            if "contraindication" in text.lower():
                clinical_info.append(f"⚠️ **Contraindications:** {self._extract_section(text, 'contraindication')}")
            if "indication" in text.lower():
                clinical_info.append(f"📋 **Indications:** {self._extract_section(text, 'indication')}")
            if "dosage" in text.lower() or "dose" in text.lower():
                clinical_info.append(f"💊 **Dosage:** {self._extract_section(text, 'dosage')}")
            if "emergency" in text.lower():
                clinical_info.append(f"🚨 **Emergency Protocol:** {self._extract_section(text, 'emergency')}")
        
        response = f"""## Clinical Decision Support

**Query:** {query}

### Evidence from MOH CPG Guidelines

{chr(10).join(clinical_info) if clinical_info else chunks[0]['text'][:600]}

### Clinical Recommendations

Based on the retrieved CPG guidelines:
- Follow standard protocols as outlined above
- Consider patient-specific factors
- Document all clinical decisions

### 📖 References
{chr(10).join(citations)}

---
*⚠️ This is an AI-assisted clinical reference. Always use clinical judgment and verify with original CPG documents.*
"""
        return response
    
    def _template_patient_response(self, query: str, chunks: List[Dict]) -> str:
        """Template-based patient education response (fallback)"""
        
        # Simplify the medical text
        medical_text = chunks[0]['text']
        simplified_text = self._simplify_medical_text(medical_text)
        
        # Extract safety information
        safety_info = ""
        if "contraindication" in medical_text.lower() or "emergency" in medical_text.lower():
            safety_info = """
### ⚠️ When to See a Healthcare Provider

- If you experience any unusual symptoms
- If your condition gets worse
- Before stopping or changing any medication
- For regular check-ups as recommended
"""
        
        response = f"""## Health Information for You

**Your question:** {query}

### What You Should Know

{simplified_text}

{safety_info}

### 💡 Key Points to Remember

- Follow your healthcare provider's advice
- Take medications exactly as prescribed
- Keep all scheduled appointments
- Ask your dentist or doctor if you have questions

---
*📌 Remember: This information is based on Malaysian health guidelines. Always talk to your healthcare provider about your specific situation.*
"""
        return response
    
    def _extract_section(self, text: str, section_type: str) -> str:
        """Extract specific section from text"""
        lines = text.split('\n')
        relevant_lines = []
        capture = False
        
        for line in lines:
            if section_type.lower() in line.lower():
                capture = True
            if capture and len(relevant_lines) < 5:
                relevant_lines.append(line.strip())
            if capture and (len(relevant_lines) >= 5 or not line.strip()):
                capture = False
        
        if relevant_lines:
            return ' '.join(relevant_lines[:3])
        return "Information available in the guidelines"
    
    def _simplify_medical_text(self, text: str) -> str:
        """Convert medical text to plain language"""
        simplified = text
        
        # Replace medical terms
        for medical, plain in self.medical_to_plain.items():
            simplified = simplified.replace(medical, plain)
            simplified = simplified.replace(medical.capitalize(), plain.capitalize())
        
        # Break long sentences
        sentences = simplified.split('.')
        simplified = '.\n'.join(sentences[:5])
        
        # Add line breaks for readability
        simplified = simplified.replace(' - ', '\n• ')
        simplified = simplified.replace(': ', ':\n• ')
        
        return simplified[:500]
    
    def _no_results_response(self, mode: str) -> str:
        """Response when no relevant information found"""
        if mode == "professional":
            return """## No Specific Guidelines Found

The query could not be matched to specific CPG recommendations.

**Suggestions:**
- Refine your search with more specific clinical terms
- Consult the full MOH CPG document library
- Consider related clinical guidelines

*For urgent clinical decisions, refer to standard protocols.*"""
        else:
            return """## Information Not Available

I couldn't find specific information about your question in the health guidelines.

**What you can do:**
- Ask your healthcare provider directly
- Try rephrasing your question
- Ask about a different health topic

*Your healthcare provider is the best source of information for your specific situation.*"""


# Singleton
_mode_handler = None

def get_mode_handler():
    global _mode_handler
    if _mode_handler is None:
        _mode_handler = ModeHandler()
    return _mode_handler