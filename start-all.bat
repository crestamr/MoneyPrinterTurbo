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

rem Set explicitly rather than trusting the inherited environment. The weights
rem are ~11GB and live on D:; without this Ollama silently falls back to
rem %USERPROFILE%\.ollama\models, finds nothing, and every LLM call fails over
rem to the raw search term instead of an expanded prompt - which looks like a
rem quality regression, not an error.
if not defined OLLAMA_MODELS set "OLLAMA_MODELS=D:\Developer\ollama-models"
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

rem Ollama and ComfyUI share the RTX 5070 Ti by taking turns rather than by
rem splitting it. OLLAMA_KEEP_ALIVE=0 unloads the model the moment a request
rem finishes, handing the card straight back to ComfyUI; reloading costs about
rem 3.6s from page cache, against 53 tok/s on the GPU versus 5 on the iGPU.
rem ComfyUI is started below with --disable-smart-memory so it releases the card
rem in the same way. Both halves are required: if either one squats on VRAM the
rem other starves, and image generation then times out entirely.
set "OLLAMA_KEEP_ALIVE=0"
start "Ollama" "%OLLAMA_EXE%" serve
call :probe_port %OLLAMA_PORT% 15
if errorlevel 1 echo ***** Ollama did not answer in time. *****

:comfyui
if "%SKIP_COMFYUI%"=="1" (
    echo ***** SKIP_COMFYUI=1 - skipping ComfyUI. *****
    goto :anime_scene
)

call :probe 1
if not errorlevel 1 (
    echo ***** ComfyUI already running at http://%COMFYUI_HOST%:%COMFYUI_PORT% *****
    goto :anime_scene
)

if not exist "%COMFYUI_DIR%\main.py" (
    echo ***** ComfyUI not found at %COMFYUI_DIR% *****
    echo ***** Set COMFYUI_DIR to your install, or pick a video source other than Qwen-Image. *****
    goto :anime_scene
)

set "COMFY_PY=python"
if exist "%COMFYUI_DIR%\venv\Scripts\python.exe" set "COMFY_PY=%COMFYUI_DIR%\venv\Scripts\python.exe"

echo ***** Starting ComfyUI at http://%COMFYUI_HOST%:%COMFYUI_PORT% (separate window) *****
rem --disable-smart-memory makes ComfyUI offload models to system RAM instead of
rem keeping them resident in VRAM, so Ollama can have the card between images.
start "ComfyUI" /d "%COMFYUI_DIR%" "%COMFY_PY%" main.py --listen %COMFYUI_HOST% --port %COMFYUI_PORT% --disable-smart-memory

echo ***** Waiting for ComfyUI to load its models (up to 2 minutes)... *****
call :probe 60
if errorlevel 1 (
    echo ***** ComfyUI did not answer in time. Check its window, then retry. *****
) else (
    echo ***** ComfyUI ready. *****
)

:anime_scene
rem Installs the MiniMax H3 "Anime Scene" workflows into ComfyUI's workflow
rem browser. Additive only: workflows edited in the UI are never overwritten.
if "%SKIP_ANIME_SCENE%"=="1" goto :openshorts
set "SETUP_PY=python"
if exist "%ROOT%.venv\Scripts\python.exe" set "SETUP_PY=%ROOT%.venv\Scripts\python.exe"
"%SETUP_PY%" "%ROOT%scripts\setup_anime_scene.py" --comfy "%COMFYUI_DIR%"
if errorlevel 1 echo ***** Anime Scene workflow setup failed - see above. *****

:openshorts
if "%SKIP_OPENSHORTS%"=="1" (
    echo ***** SKIP_OPENSHORTS=1 - skipping OpenShorts. *****
    goto :webui
)
if not defined OPENSHORTS_DIR set "OPENSHORTS_DIR=D:\Developer\openshorts"
if not exist "%OPENSHORTS_DIR%\start-openshorts.bat" (
    echo ***** OpenShorts launcher not found in %OPENSHORTS_DIR% - skipping. *****
    goto :webui
)
rem Own window: right after a Podman VM restart OpenShorts can take ~3 minutes
rem to answer, which must not hold up the MoneyPrinterTurbo WebUI below.
echo ***** Starting OpenShorts at http://localhost:5175 (separate window) *****
start "OpenShorts" /min cmd /c ""%OPENSHORTS_DIR%\start-openshorts.bat""

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
