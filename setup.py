"""
Setup Wizard - Complete setup of RAG Chatbot
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def check_python_version():
    """Check if Python version is 3.8+"""
    print_header("1. Checking Python Version")
    
    version = sys.version_info
    print(f"Current Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version OK")
    return True


def create_venv():
    """Create virtual environment"""
    print_header("2. Creating Virtual Environment")
    
    if os.path.exists("venv"):
        print("✅ Virtual environment already exists")
        return True
    
    print("Creating venv...")
    result = subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    if result.returncode == 0:
        print("✅ Virtual environment created")
        return True
    else:
        print("❌ Failed to create virtual environment")
        return False


def activate_venv():
    """Activate virtual environment"""
    print_header("3. Activating Virtual Environment")
    
    if sys.platform == "win32":
        venv_activate = "venv\\Scripts\\activate.bat"
        print(f"On Windows, run: {venv_activate}")
    else:
        venv_activate = "source venv/bin/activate"
        print(f"On Linux/Mac, run: {venv_activate}")
    
    print("\n⚠️ Please activate the virtual environment manually")
    return True


def install_dependencies():
    """Install Python packages"""
    print_header("4. Installing Dependencies")
    
    print("Installing packages (this may take 1-2 minutes)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True
    )
    
    if result.returncode == 0:
        print("✅ Dependencies installed")
        return True
    else:
        print("❌ Installation failed")
        print(result.stderr.decode())
        return False


def setup_env_file():
    """Create .env file"""
    print_header("5. Setting Up Environment Variables")
    
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    # Copy from example
    if os.path.exists(".env.example"):
        with open(".env.example", "r") as f:
            content = f.read()
        
        with open(".env", "w") as f:
            f.write(content)
        
        print("✅ .env file created from template")
    else:
        # Create new .env
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=your_api_key_here\n")
            f.write("MODEL=gpt-3.5-turbo\n")
        
        print("✅ .env file created")
    
    return True


def edit_env_file():
    """Prompt user to edit .env file"""
    print_header("6. Configuring OpenAI API Key")
    
    print("⚠️ IMPORTANT: You need an OpenAI API key")
    print("\n1. Go to: https://platform.openai.com/api-keys")
    print("2. Create a new API key")
    print("3. Edit .env file and add your key:")
    print("   OPENAI_API_KEY=sk-...")
    
    input("\n📌 Press Enter once you've added your API key to .env")
    
    # Verify key exists
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            content = f.read()
            if "your_api_key_here" in content:
                print("⚠️ API key still not set")
                return False
            else:
                print("✅ API key configured")
                return True
    
    return False


def test_installation():
    """Test if everything works"""
    print_header("7. Testing Installation")
    
    try:
        print("Testing imports...")
        from rag_pipeline import RAGPipeline
        from advanced_features import ChatMemory
        print("✅ All imports successful")
        
        print("\nTesting environment variables...")
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_api_key_here":
            print("✅ OpenAI API key found")
        else:
            print("❌ OpenAI API key not configured")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_app():
    """Run the Streamlit app"""
    print_header("8. Running Application")
    
    print("🚀 Starting Streamlit app...")
    print("\nThe app will open at: http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    
    os.system("streamlit run app.py")


def show_next_steps():
    """Show next steps"""
    print_header("Setup Complete! 🎉")
    
    print("""
✅ All setup steps completed!

📌 Next Steps:

1. 📄 Upload PDFs or Websites
   - Go to "Upload & Learn" tab
   - Upload PDF files or enter website URLs
   
2. 💬 Ask Questions
   - Go to "Chat" tab
   - Type your questions
   - Get AI-powered answers with source citations
   
3. 🚀 Advanced Features
   - Save/load vector databases
   - Hybrid search
   - Conversation history
   - Role-based AI

📚 Documentation:
   - README.md - Full documentation
   - examples.py - Usage examples
   - DEPLOYMENT.md - Deployment guide

💡 Tips:
   - Start with small PDFs for testing
   - Adjust chunk size based on content
   - Use gpt-3.5-turbo for fast responses
   - Save your vector DB to avoid reprocessing

🆘 Troubleshooting:
   - Check .env file has OPENAI_API_KEY
   - Ensure all dependencies installed
   - Check internet connection
   - Monitor API usage/costs

Good luck! 🚀
    """)


def main():
    """Main setup wizard"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " 🤖 RAG Chatbot - Setup Wizard ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    steps = [
        ("Python Version", check_python_version),
        ("Virtual Environment", create_venv),
        ("Install Dependencies", install_dependencies),
        ("Setup .env File", setup_env_file),
        ("Configure API Key", edit_env_file),
        ("Test Installation", test_installation),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please fix the issue and run setup again")
            return False
    
    show_next_steps()
    
    # Ask if user wants to run app
    response = input("\n🚀 Run Streamlit app now? (yes/no): ").lower()
    if response in ["yes", "y"]:
        run_app()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
