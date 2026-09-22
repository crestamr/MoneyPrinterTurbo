@echo off
setlocal
set "ROOT=%~dp0"
echo ***** Current directory: %ROOT% *****
set "PYTHONPATH=%ROOT%;%PYTHONPATH%"

rem If you cannot download the model weights from the official site, use a mirror.
rem Just remove the "rem" from the following line.
rem set HF_ENDPOINT=https://hf-mirror.com

rem OmniVoice needs torch, which the main app venv deliberately does not install.
rem Point OMNIVOICE_PYTHON at the interpreter of the environment you installed
rem services\omnivoice_server\requirements.txt into. See that folder's README.md.
if defined OMNIVOICE_PYTHON (
    set "PY=%OMNIVOICE_PYTHON%"
) else if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PY=%ROOT%.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo ***** Starting OmniVoice TTS service on http://127.0.0.1:8890 *****
"%PY%" -m services.omnivoice_server %*
endlocal
