"""
Setup script for Ollama RAG Chatbot
Downloads and prepares Ollama and models
"""

import subprocess
import sys
import requests
import time
import platform
from pathlib import Path


def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"✅ Ollama is installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False


def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
    except requests.exceptions.RequestException:
        print("❌ Ollama service is not running")
        return False


def install_ollama():
    """Install Ollama"""
    print("\n📥 Installing Ollama...")
    
    if platform.system() == "Windows":
        print("Downloading Ollama for Windows...")
        url = "https://ollama.ai/download/OllamaSetup.exe"
        installer_path = Path("OllamaSetup.exe")
        
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        
        with open(installer_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Running installer...")
        subprocess.Popen(str(installer_path))
        print("Installer launched. Please follow the installation steps.")
        print("After installation, restart this script.")
        
    elif platform.system() == "Darwin":  # macOS
        print("Installing Ollama via Homebrew...")
        subprocess.run(["brew", "install", "ollama"], check=True)
        
    elif platform.system() == "Linux":
        print("Installing Ollama via curl...")
        subprocess.run(["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"], 
                      shell=True, check=True)


def start_ollama_service():
    """Start Ollama service (Linux/Mac only, Windows runs as service)"""
    if platform.system() in ["Linux", "Darwin"]:
        print("\n▶️  Starting Ollama service...")
        try:
            subprocess.Popen(["ollama", "serve"])
            time.sleep(3)
            if check_ollama_running():
                return True
        except Exception as e:
            print(f"❌ Failed to start Ollama: {e}")
            return False
    else:
        print("⏳ Waiting for Ollama service to start...")
        for i in range(30):
            if check_ollama_running():
                return True
            time.sleep(1)
        return False


def pull_model(model_name: str = "mistral"):
    """Pull an Ollama model"""
    print(f"\n📥 Pulling model: {model_name}...")
    
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print(f"✅ Model {model_name} pulled successfully")
            return True
        else:
            print(f"❌ Failed to pull model: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Model pull timed out")
        return False
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False


def setup_environment():
    """Setup Python environment"""
    print("\n📦 Installing Python dependencies...")
    
    requirements = [
        "langchain-ollama",
        "langchain-community",
        "langchain",
        "faiss-cpu",
        "streamlit",
        "python-dotenv",
        "requests",
    ]
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + requirements,
            check=True
        )
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def main():
    """Main setup flow"""
    print("=" * 60)
    print("🚀 Ollama RAG Chatbot Setup")
    print("=" * 60)
    
    # Step 1: Check/Install Ollama
    if not check_ollama_installed():
        response = input("\nOllama not found. Install it? (y/n): ").lower()
        if response == 'y':
            install_ollama()
            return  # Exit and wait for manual restart
        else:
            print("❌ Ollama is required")
            sys.exit(1)
    
    # Step 2: Setup Python environment
    if not setup_environment():
        sys.exit(1)
    
    # Step 3: Start Ollama service
    if not start_ollama_service():
        print("\n⚠️  Ollama service is not running")
        print("On Windows: Ollama should start automatically")
        print("On Mac/Linux: Run 'ollama serve' in another terminal")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 4: Pull model
    print("\n🤖 Available models:")
    print("  - mistral (7B, fast, good quality)")
    print("  - neural-chat (7B, conversational)")
    print("  - llama2 (7B, general purpose)")
    print("  - dolphin-mixtral (8x7B, powerful)")
    
    model = input("\nWhich model to download? (default: mistral): ").strip() or "mistral"
    
    if pull_model(model):
        print("\n✅ Setup complete!")
        print(f"\n📝 Update .env file with: OLLAMA_MODEL={model}")
        print("\n▶️  To start the app, run:")
        print("   streamlit run app_ollama.py")
    else:
        print("\n⚠️  Model pull failed, but you can try manually:")
        print(f"   ollama pull {model}")


if __name__ == "__main__":
    main()
