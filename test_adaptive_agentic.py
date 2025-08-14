#!/usr/bin/env python3
"""
Test script to verify that adaptive chunking now includes agentic chunking
"""

import requests
import json
import time

# Test the enhanced adaptive chunking with a complex document
def test_adaptive_with_agentic():
    # Create a complex document that should trigger agentic chunking
    complex_document = """
    ARTIFICIAL INTELLIGENCE RESEARCH METHODOLOGY AND IMPLEMENTATION FRAMEWORK
    
    Abstract: This research paper presents a comprehensive methodology for implementing artificial intelligence systems in enterprise environments, analyzing the correlation between various machine learning algorithms and their statistical significance in real-world applications.
    
    1. Introduction
    The implementation of artificial intelligence (AI) frameworks requires a systematic approach to algorithm selection, data preprocessing, and model validation. According to recent studies (Smith et al., 2023), the effectiveness of AI systems depends significantly on the methodology employed during the development phase.
    
    2. Technical Architecture and Protocol Specifications
    2.1 Database Schema Design
    The proposed architecture utilizes a multi-tier database schema with the following specifications:
    - Primary tables for entity storage
    - Secondary indices for performance optimization
    - Foreign key constraints for data integrity
    - JSON document storage for flexible metadata
    
    2.2 API Endpoint Configuration
    REST API endpoints must comply with the following requirements:
    - HTTP/HTTPS protocol support
    - JSON/XML data format compatibility
    - Authentication token validation
    - Rate limiting implementation
    
    3. Legal and Compliance Framework
    Whereas the implementation of AI systems must comply with relevant regulations, the following provisions shall be considered:
    
    3.1 Data Privacy Requirements
    All parties must ensure that data processing activities comply with applicable privacy laws. The organization shall implement appropriate technical and organizational measures to protect personal data pursuant to Article 32 of the GDPR.
    
    3.2 Liability and Indemnification
    Each party agrees to indemnify and hold harmless the other party from any claims arising from the use of the AI system, notwithstanding any limitations set forth herein.
    
    4. Research Methodology and Statistical Analysis
    4.1 Hypothesis Testing
    The research hypothesis states that adaptive algorithms demonstrate statistically significant improvements over traditional approaches (p-value < 0.05). Variables were controlled for confounding factors, and correlation analysis was performed using Pearson's correlation coefficient.
    
    4.2 Literature Review
    Previous studies have shown that machine learning models exhibit varying performance across different domains. The meta-analysis conducted by Johnson et al. (2022) revealed significant heterogeneity in effect sizes (I² = 78%, p < 0.001).
    
    5. Implementation Guidelines and Best Practices
    5.1 Code Architecture
    The following code structure shall be implemented:
    ```python
    class AIFramework:
        def __init__(self, config):
            self.model = None
            self.preprocessor = None
        
        def train(self, data):
            # Implementation details
            pass
    ```
    
    5.2 Performance Optimization
    - Utilize vectorized operations for mathematical computations
    - Implement caching mechanisms for frequently accessed data
    - Apply parallel processing where applicable
    - Monitor memory usage and optimize accordingly
    
    6. Conclusion
    This comprehensive framework provides a robust foundation for AI implementation in enterprise environments. The methodology presented herein demonstrates the importance of systematic approaches to algorithm selection and validation.
    
    References:
    1. Smith, J., et al. (2023). "Advanced AI Methodologies." Journal of Artificial Intelligence Research, 45(2), 123-145.
    2. Johnson, M., et al. (2022). "Meta-analysis of Machine Learning Performance." Nature Machine Intelligence, 4, 567-580.
    """
    
    # Test adaptive chunking
    print("Testing Enhanced Adaptive Chunking with Complex Document...")
    print("=" * 60)
    
    # Send request to backend
    url = "http://localhost:5001/api/test-chunking"
    
    test_data = {
        "text": complex_document,
        "method": "adaptive",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Request successful!")
            print(f"📊 Number of chunks created: {len(result.get('chunks', []))}")
            
            # Check which method was selected by adaptive chunking
            if result.get('chunks'):
                first_chunk = result['chunks'][0]
                metadata = first_chunk.get('metadata', {})
                selected_method = metadata.get('adaptive_method', 'unknown')
                text_analysis = metadata.get('text_analysis', {})
                
                print(f"🎯 Adaptive method selected: {selected_method}")
                print(f"📈 Document complexity score: {text_analysis.get('complexity_score', 'N/A')}")
                print(f"🔍 Document analysis:")
                print(f"   - Length: {text_analysis.get('length', 'N/A')} characters")
                print(f"   - Technical indicators: {text_analysis.get('technical_indicators', 'N/A')}")
                print(f"   - Legal indicators: {text_analysis.get('legal_indicators', 'N/A')}")
                print(f"   - Academic indicators: {text_analysis.get('academic_indicators', 'N/A')}")
                print(f"   - Is complex: {text_analysis.get('is_complex', 'N/A')}")
                print(f"   - Needs intelligent chunking: {text_analysis.get('needs_intelligent_chunking', 'N/A')}")
                
                if selected_method == 'agentic':
                    print("🎉 SUCCESS: Adaptive method correctly selected AGENTIC chunking!")
                    print("✨ This confirms that agentic chunking has been successfully integrated into adaptive method.")
                elif selected_method in ['semantic', 'recursive', 'sentence']:
                    print(f"📝 INFO: Adaptive method selected {selected_method} chunking.")
                    print("💡 This might be due to API availability or document characteristics.")
                else:
                    print(f"⚠️  UNKNOWN: Unexpected method selected: {selected_method}")
                
                print("\n📄 Sample chunk preview:")
                chunk_content = first_chunk.get('content', '')[:200]
                print(f"   {chunk_content}{'...' if len(chunk_content) >= 200 else ''}")
                
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the backend is running on http://localhost:5001")

if __name__ == "__main__":
    test_adaptive_with_agentic()
