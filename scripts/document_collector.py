"""
Document Collection and Preparation Module
For MOH CPG Oral Health Guidelines
"""

import os
import json
import pdfplumber
from typing import Dict, List
from datetime import datetime

class DocumentCollector:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.document_registry = []
        
    def register_document(self, title: str, category: str, year: int, 
                          target_mode: str, source_file: str):
        """Register a document in the tracking system"""
        
        doc_info = {
            "title": title,
            "category": category,
            "year": year,
            "target_mode": target_mode,  # "professional", "patient", or "both"
            "source_file": source_file,
            "status": "pending",
            "collected_date": None,
            "chunks_count": 0
        }
        self.document_registry.append(doc_info)
        return doc_info
    
    def extract_document_info(self, pdf_path: str) -> Dict:
        """Extract metadata from PDF document"""
        
        with pdfplumber.open(pdf_path) as pdf:
            # Extract first page for title detection
            first_page = pdf.pages[0].extract_text()
            
            # Try to detect document type from content
            doc_info = {
                "page_count": len(pdf.pages),
                "has_headers": self._detect_headers(first_page),
                "has_sections": self._detect_sections(first_page),
                "extraction_date": datetime.now().isoformat()
            }
            
            return doc_info
    
    def _detect_headers(self, text: str) -> List[str]:
        """Detect section headers in document"""
        headers = []
        lines = text.split('\n')
        for line in lines:
            # Look for common header patterns
            if (line.isupper() and len(line) > 10) or \
               (line.endswith(':') and len(line) < 50) or \
               (line.startswith('SECTION') or line.startswith('CHAPTER')):
                headers.append(line.strip())
        return headers[:10]  # Return first 10 headers
    
    def _detect_sections(self, text: str) -> List[str]:
        """Detect clinical sections in document"""
        clinical_sections = [
            "INDICATIONS", "CONTRAINDICATIONS", "DOSAGE", 
            "MANAGEMENT", "TREATMENT", "PROCEDURE", "ASSESSMENT",
            "DIAGNOSIS", "PREVENTION", "REFERRAL", "FOLLOW-UP"
        ]
        
        found_sections = []
        text_upper = text.upper()
        for section in clinical_sections:
            if section in text_upper:
                found_sections.append(section)
        
        return found_sections
    
    def generate_collection_report(self) -> str:
        """Generate report of collected documents"""
        
        report = f"""
========================================
CPG DOCUMENT COLLECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================

Total Documents Registered: {len(self.document_registry)}

By Target Mode:
- Professional Mode: {len([d for d in self.document_registry if d['target_mode'] in ['professional', 'both']])}
- Patient Mode: {len([d for d in self.document_registry if d['target_mode'] in ['patient', 'both']])}



By Category:
"""
        # Group by category
        categories = {}
        for doc in self.document_registry:
            cat = doc['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            report += f"  - {cat}: {count}\n"
        
        report += "\nDocument Details:\n"
        report += "-" * 40 + "\n"
        
        for doc in self.document_registry:
            report += f"""
📄 {doc['title']}
   Category: {doc['category']} | Year: {doc['year']}
   Mode: {doc['target_mode']} | Status: {doc['status']}
   Source: {doc['source_file']}
"""
        
        return report
    
    def save_registry(self, output_file="document_registry.json"):
        """Save document registry to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.document_registry, f, indent=2)
        print(f"✅ Document registry saved to {output_file}")

# Example usage
if __name__ == "__main__":
    collector = DocumentCollector()
    
    # Register all your CPG documents
    documents = [
        # Professional Mode Documents
        ("Clinical Prevention Programme for Caries (CPPC)", "Prevention", 2024, "professional", "CPPC_2024.pdf"),
        ("Primary Prevention and Early Detection of Oral Cancer", "Oncology", 2023, "professional", "Oral_Cancer_2023.pdf"),
        ("Periodontal and Peri-Implant Disease Management", "Periodontics", 2024, "professional", "Periodontal_2024.pdf"),
        ("Orthodontic Treatment Referral Guidelines", "Orthodontics", 2023, "professional", "Orthodontic_2023.pdf"),
        ("Modified MOH ICDAS and Caries Risk Assessment", "Diagnosis", 2024, "both", "ICDAS_2024.pdf"),
        
        # Patient Mode Documents
        ("Effective Toothbrushing Training SOP", "Patient Education", 2024, "patient", "Toothbrushing_SOP.pdf"),
        ("Oral Healthcare for Preschool Children", "Pediatrics", 2023, "patient", "Preschool_Oral_2023.pdf"),
        ("Oral Healthcare for Pregnant Mothers", "Prenatal", 2024, "patient", "Pregnant_Oral_2024.pdf"),
        ("Oral Healthcare for Elderly Population", "Geriatrics", 2024, "patient", "Elderly_Oral_2024.pdf"),
        ("Tobacco-Free Oral Health Programme Guidelines", "Tobacco Cessation", 2023, "patient", "Tobacco_Free_2023.pdf")
    ]
    
    for doc in documents:
        collector.register_document(*doc)
    
    # Save registry
    collector.save_registry()
    
    # Generate report
    print(collector.generate_collection_report())
