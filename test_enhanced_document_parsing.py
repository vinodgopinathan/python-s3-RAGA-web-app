#!/usr/bin/env python3
"""
Test enhanced document parsing capabilities
"""

import sys
import os

# Add the backend source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

def test_library_imports():
    """Test if all required libraries are available"""
    print("🔍 Testing library imports...")
    
    # Test basic imports
    try:
        import PyPDF2
        print("✅ PyPDF2 available")
    except ImportError as e:
        print(f"❌ PyPDF2 not available: {e}")
    
    # Test Tesseract OCR
    try:
        import pytesseract
        from PIL import Image
        print("✅ Tesseract OCR (pytesseract + Pillow) available")
        
        # Test if tesseract binary is available
        try:
            pytesseract.get_tesseract_version()
            print("✅ Tesseract binary accessible")
        except Exception as e:
            print(f"⚠️ Tesseract binary issue: {e}")
    except ImportError as e:
        print(f"❌ Tesseract OCR not available: {e}")
    
    # Test Excel processing
    try:
        import xlrd
        print("✅ xlrd (Excel .xls) available")
    except ImportError as e:
        print(f"❌ xlrd not available: {e}")
    
    try:
        import openpyxl
        print("✅ openpyxl (Excel .xlsx) available")
    except ImportError as e:
        print(f"❌ openpyxl not available: {e}")
    
    # Test Word processing
    try:
        from docx import Document
        print("✅ python-docx (Word documents) available")
    except ImportError as e:
        print(f"❌ python-docx not available: {e}")

def test_s3_helper():
    """Test S3Helper enhanced functionality"""
    print("\n🔍 Testing S3Helper enhanced functionality...")
    
    try:
        from utils.s3_helper import S3Helper
        
        # Create S3Helper instance (this will require AWS credentials in production)
        print("Creating S3Helper instance...")
        s3_helper = S3Helper()
        
        # Test supported file types
        supported_types = s3_helper.get_supported_file_types()
        print(f"✅ Supported file types: {supported_types}")
        
        total_extensions = sum(len(exts) for exts in supported_types.values())
        print(f"📊 Total supported extensions: {total_extensions}")
        
    except Exception as e:
        print(f"❌ Error testing S3Helper: {e}")

def test_document_processor():
    """Test DocumentProcessor with new file types"""
    print("\n🔍 Testing DocumentProcessor...")
    
    try:
        from utils.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("✅ DocumentProcessor created successfully")
        
        # Test file type detection
        test_files = [
            "document.pdf",
            "image.jpg", 
            "spreadsheet.xlsx",
            "legacy_sheet.xls",
            "report.docx",
            "text.txt"
        ]
        
        print("\n📝 File type detection tests:")
        for file_name in test_files:
            file_type = processor.get_file_type(file_name)
            print(f"  {file_name} -> {file_type}")
        
    except Exception as e:
        print(f"❌ Error testing DocumentProcessor: {e}")

def main():
    """Run all tests"""
    print("🚀 Enhanced Document Parsing Test Suite")
    print("=" * 50)
    
    test_library_imports()
    test_s3_helper()
    test_document_processor()
    
    print("\n✅ Test suite completed!")
    print("\n💡 Note: AWS credentials are required for full S3Helper functionality")
    print("💡 In Docker, tesseract-ocr package must be installed at system level")

if __name__ == "__main__":
    main()
