#!/usr/bin/env python3
import os
import sys
import logging

# Add the backend src to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from utils.document_processor import DocumentProcessor

# Configure logging to see detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_agentic_chunking():
    """Test agentic chunking directly with various text samples"""
    
    # Create processor instance
    processor = DocumentProcessor()
    
    # Test 1: Simple text
    print("\n" + "="*80)
    print("TEST 1: Simple Text")
    print("="*80)
    
    simple_text = """
    This is a simple test document. It has multiple sentences that should be chunked intelligently.
    
    The agentic chunking system should analyze this text and determine optimal boundaries. Let's see if it works correctly.
    
    This is another paragraph with different content. It talks about testing and validation of the chunking system.
    """
    
    # Set to agentic chunking
    processor.set_chunking_method('agentic')
    
    try:
        chunks = processor.chunk_text(simple_text)
        print(f"Number of chunks created: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i + 1}:")
            print(f"Method: {chunk['metadata'].get('chunking_method', 'unknown')}")
            print(f"Content: {chunk['content'][:100]}...")
            print(f"Metadata: {chunk['metadata']}")
            
    except Exception as e:
        logger.error(f"Error in simple text test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Complex organizational text (similar to camp documents)
    print("\n" + "="*80)
    print("TEST 2: Complex Organizational Text")
    print("="*80)
    
    complex_text = """
    Camp Wildwood Staff Directory
    
    Director: John Smith
    Email: john@campwildwood.org
    Phone: (555) 123-4567
    
    Assistant Director: Mary Johnson
    Email: mary@campwildwood.org
    Phone: (555) 234-5678
    
    Program Coordinator: Bob Wilson
    Responsible for activity planning and execution.
    Background: 10 years experience in youth programs.
    
    Waterfront Director: Lisa Davis
    Certified in water safety and rescue operations.
    Manages all swimming and boating activities.
    
    Camp Operations
    
    The camp operates from June through August each year.
    We serve children ages 6-16 with various programs.
    
    Safety Protocols
    
    All staff must complete safety training before the season begins.
    Emergency procedures are reviewed weekly with all personnel.
    """
    
    try:
        chunks = processor.chunk_text(complex_text)
        print(f"Number of chunks created: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i + 1}:")
            print(f"Method: {chunk['metadata'].get('chunking_method', 'unknown')}")
            print(f"Content: {chunk['content'][:150]}...")
            print(f"LLM Reasoning: {chunk['metadata'].get('llm_reasoning', 'N/A')}")
            print(f"Confidence: {chunk['metadata'].get('llm_confidence', 'N/A')}")
            
    except Exception as e:
        logger.error(f"Error in complex text test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check environment variables
    print("\n" + "="*80)
    print("TEST 3: Environment Check")
    print("="*80)
    
    print(f"GEMINI_API_KEY set: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
    print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'Not set')}")
    print(f"Processor chunking method: {processor.chunking_method}")
    print(f"Processor LLM provider: {processor.llm_provider}")
    print(f"Processor Gemini API key set: {'Yes' if processor.gemini_api_key else 'No'}")
    
    # Test 4: Direct Gemini API test
    print("\n" + "="*80)
    print("TEST 4: Direct Gemini API Test")
    print("="*80)
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content("Hello, can you respond with a simple JSON object containing a greeting?")
        print(f"Gemini API response: {response.text}")
        
    except Exception as e:
        logger.error(f"Error testing Gemini API directly: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agentic_chunking()
