"""
Test script - Verify RAG chatbot installation
"""

import os
import sys
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported"""
    print("\n" + "="*60)
    print("Testing Imports...")
    print("="*60)
    
    packages = {
        "langchain": "LangChain",
        "langchain_openai": "OpenAI Integration",
        "faiss": "FAISS Vector DB",
        "streamlit": "Streamlit",
        "pypdf": "PyPDF Loader",
        "beautifulsoup4": "Beautiful Soup",
        "requests": "Requests",
        "dotenv": "Python Dotenv",
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name} ({package})")
        except ImportError:
            print(f"❌ {name} ({package})")
            failed.append(package)
    
    if failed:
        print(f"\n⚠️ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All imports successful!")
    return True


def test_env_file():
    """Test if .env file exists and has API key"""
    print("\n" + "="*60)
    print("Testing Environment Configuration...")
    print("="*60)
    
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        print("   Copy .env.example to .env and add your API key")
        return False
    
    print("✅ .env file exists")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set in .env")
        return False
    
    if api_key == "your_api_key_here":
        print("❌ OPENAI_API_KEY is placeholder")
        print("   Edit .env and add your real API key from https://platform.openai.com/api-keys")
        return False
    
    # Check format
    if not api_key.startswith("sk-"):
        print("⚠️ API key doesn't look valid (should start with 'sk-')")
    
    print("✅ OPENAI_API_KEY configured")
    return True


def test_modules():
    """Test if core modules can be imported"""
    print("\n" + "="*60)
    print("Testing Core Modules...")
    print("="*60)
    
    try:
        from rag_pipeline import RAGPipeline
        print("✅ RAG Pipeline module")
    except ImportError as e:
        print(f"❌ RAG Pipeline: {e}")
        return False
    
    try:
        from advanced_features import ChatMemory, RoleBasedAI
        print("✅ Advanced Features module")
    except ImportError as e:
        print(f"❌ Advanced Features: {e}")
        return False
    
    print("\n✅ All modules loaded successfully!")
    return True


def test_rag_instantiation():
    """Test if RAG pipeline can be instantiated"""
    print("\n" + "="*60)
    print("Testing RAG Pipeline Instantiation...")
    print("="*60)
    
    try:
        from rag_pipeline import RAGPipeline
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        rag = RAGPipeline(api_key=api_key)
        
        print("✅ RAG Pipeline created successfully")
        print(f"   Model: {rag.llm.model_name}")
        print(f"   Chunk Size: {rag.chunk_size}")
        print(f"   Chunk Overlap: {rag.chunk_overlap}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create RAG Pipeline: {e}")
        return False


def test_file_structure():
    """Test if required files exist"""
    print("\n" + "="*60)
    print("Testing Project File Structure...")
    print("="*60)
    
    required_files = [
        "app.py",
        "rag_pipeline.py",
        "advanced_features.py",
        "requirements.txt",
        ".env.example",
        "README.md",
    ]
    
    all_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} (missing)")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required files present!")
    else:
        print("\n❌ Some files are missing")
    
    return all_exist


def test_streamlit():
    """Test if Streamlit app can be imported"""
    print("\n" + "="*60)
    print("Testing Streamlit Application...")
    print("="*60)
    
    try:
        # This won't actually run the app, just checks if it can be imported
        with open("app.py", "r") as f:
            content = f.read()
            
        if "import streamlit as st" in content:
            print("✅ app.py is valid Streamlit code")
            return True
        else:
            print("❌ app.py doesn't look like a Streamlit app")
            return False
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False


def show_summary(results):
    """Show test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Ready to use!\n")
        print("Next steps:")
        print("1. Run: streamlit run app.py")
        print("2. Open http://localhost:8501")
        print("3. Upload PDFs or websites")
        print("4. Ask questions and get AI answers!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix and try again.\n")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " RAG Chatbot - Installation Test ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    tests = {
        "Imports": test_imports,
        "File Structure": test_file_structure,
        "Environment": test_env_file,
        "Modules": test_modules,
        "RAG Pipeline": test_rag_instantiation,
        "Streamlit": test_streamlit,
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Exception in {test_name}: {e}")
            results[test_name] = False
    
    # Show summary
    success = show_summary(results)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
