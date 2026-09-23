@echo off
setlocal
set "ROOT=%~dp0"

rem ComfyUI serves the Qwen-Image 2.1 video source. Everything else in the app
rem runs without it, so a missing install is a warning, not a failure.
if not defined COMFYUI_DIR  set "COMFYUI_DIR=D:\Developer\ComfyUI"
if not defined COMFYUI_HOST set "COMFYUI_HOST=127.0.0.1"
if not defined COMFYUI_PORT set "COMFYUI_PORT=8188"

rem Ollama serves the local LLM that writes scripts and expands image prompts.
rem Its own tray app usually starts it at login, so this is just a safety net.
if not defined OLLAMA_PORT set "OLLAMA_PORT=11434"
if "%SKIP_OLLAMA%"=="1" goto :comfyui

call :probe_port %OLLAMA_PORT% 1
if not errorlevel 1 (
    echo ***** Ollama already running at http://127.0.0.1:%OLLAMA_PORT% *****
    goto :comfyui
)

set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not exist "%OLLAMA_EXE%" (
    echo ***** Ollama not found - the LLM provider will be unavailable. *****
    goto :comfyui
)

echo ***** Starting Ollama at http://127.0.0.1:%OLLAMA_PORT% (separate window) *****
if not defined OLLAMA_HOST set "OLLAMA_HOST=127.0.0.1:%OLLAMA_PORT%"
start "Ollama" "%OLLAMA_EXE%" serve
call :probe_port %OLLAMA_PORT% 15
if errorlevel 1 echo ***** Ollama did not answer in time. *****

:comfyui
if "%SKIP_COMFYUI%"=="1" (
    echo ***** SKIP_COMFYUI=1 - skipping ComfyUI. *****
    goto :webui
)

call :probe 1
if not errorlevel 1 (
    echo ***** ComfyUI already running at http://%COMFYUI_HOST%:%COMFYUI_PORT% *****
    goto :webui
)

if not exist "%COMFYUI_DIR%\main.py" (
    echo ***** ComfyUI not found at %COMFYUI_DIR% *****
    echo ***** Set COMFYUI_DIR to your install, or pick a video source other than Qwen-Image. *****
    goto :webui
)

set "COMFY_PY=python"
if exist "%COMFYUI_DIR%\venv\Scripts\python.exe" set "COMFY_PY=%COMFYUI_DIR%\venv\Scripts\python.exe"

echo ***** Starting ComfyUI at http://%COMFYUI_HOST%:%COMFYUI_PORT% (separate window) *****
start "ComfyUI" /d "%COMFYUI_DIR%" "%COMFY_PY%" main.py --listen %COMFYUI_HOST% --port %COMFYUI_PORT%

echo ***** Waiting for ComfyUI to load its models (up to 2 minutes)... *****
call :probe 60
if errorlevel 1 (
    echo ***** ComfyUI did not answer in time. Check its window, then retry. *****
) else (
    echo ***** ComfyUI ready. *****
)

:webui
echo.
pushd "%ROOT%"
call "%ROOT%webui.bat"
popd
endlocal
exit /b 0

rem :probe <attempts> - polls ComfyUI every 2s, exits 0 as soon as it answers.
:probe
powershell -NoProfile -ExecutionPolicy Bypass -Command "$n=[int]'%1'; for ($i=0; $i -lt $n; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 ('http://' + $env:COMFYUI_HOST + ':' + $env:COMFYUI_PORT + '/system_stats') | Out-Null; exit 0 } catch { if ($i -lt $n - 1) { Start-Sleep -Seconds 2 } } }; exit 1"
exit /b %errorlevel%

rem :probe_port <port> <attempts> - polls a local TCP port every 2s.
:probe_port
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=[int]'%1'; $n=[int]'%2'; for ($i=0; $i -lt $n; $i++) { $c=New-Object Net.Sockets.TcpClient; try { $c.Connect('127.0.0.1',$p); $c.Close(); exit 0 } catch { try { $c.Close() } catch {}; if ($i -lt $n - 1) { Start-Sleep -Seconds 2 } } }; exit 1"
exit /b %errorlevel%
