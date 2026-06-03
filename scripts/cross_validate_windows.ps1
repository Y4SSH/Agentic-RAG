<#
Cross-machine validation script for Windows (PowerShell).
Run on a clean Windows machine from the repo root.

Steps performed:
 - Create / activate venv
 - Install requirements
 - Ensure `ollama` is available
 - Pull `tinyllama` into Ollama
 - Set `.env` OLLAMA_MODEL_NAME to `tinyllama`
 - Start Streamlit (manual step shown)
 - Run the smoke-test script

Usage (PowerShell):
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    ./scripts/cross_validate_windows.ps1
#>

Write-Host "Starting cross-machine validation script..."

# 1) Create venv if missing
if (-Not (Test-Path .venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# 2) Activate venv for the remainder of the script (user may need to permit script execution)
Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

# 3) Install dependencies
Write-Host "Installing Python dependencies (this may take a few minutes)..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4) Ensure Ollama present
if (-Not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'ollama' was not found in PATH. Please install Ollama from https://ollama.ai and re-run this script." -ForegroundColor Red
    exit 2
}

# 5) Pull model
Write-Host "Pulling tinyllama model into Ollama (may take several minutes)..."
ollama pull tinyllama

# 6) Ensure .env contains the proper model
Write-Host "Writing .env with OLLAMA_MODEL_NAME=tinyllama"
Set-Content -Path .env -Value "OLLAMA_MODEL_NAME=tinyllama"

Write-Host "Now start the Streamlit app in a separate terminal with:"
Write-Host "    python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501"
Write-Host "After Streamlit is running, you can run the smoke tests:"
Write-Host "    .venv\Scripts\python.exe scripts\run_smoke_tests.py"

Write-Host "Cross-machine validation script completed. Follow the printed instructions to finish validation."