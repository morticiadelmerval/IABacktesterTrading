@echo off
echo ========================================================
echo Instalador Cuantitativo con Soporte NVIDIA (CUDA 12)
echo ========================================================
echo.

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Acelerador 'uv' detectado. La instalacion sera ultra-rapida.
    set VENV_CMD=uv venv .venv
    set PIP_CMD=uv pip install
) else (
    echo [INFO] 'uv' no detectado. Usando Python estandar (puede ser lento).
    set VENV_CMD=python -m venv .venv
    set PIP_CMD=pip install
)

echo.
echo Paso 1: Creando entorno virtual local (.venv)...
if not exist ".venv" (
    %VENV_CMD%
)
call .venv\Scripts\activate.bat

echo.
echo Paso 2: Instalando dependencias base...
%PIP_CMD% yfinance pandas numpy scikit-learn

echo.
echo Paso 3: Instalando ecosistema de IA...
%PIP_CMD% transformers accelerate huggingface-hub

echo.
echo Paso 4: Instalando PyTorch optimizado para NVIDIA (Serie 3000/4000)...
echo Descargando binarios CUDA 12.1+ (Esto tomara varios minutos...)
%PIP_CMD% torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo ========================================================
echo INSTALACION COMPLETADA EXITOSAMENTE
echo ========================================================
echo.
echo Para correr el oraculo de IA (TimesFM):
echo   .venv\Scripts\python precalculate_timesfm.py
echo.
echo Para arrancar el Dashboard web:
echo   .venv\Scripts\python server.py
echo.
pause
