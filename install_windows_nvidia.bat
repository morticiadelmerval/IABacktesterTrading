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
%PIP_CMD% transformers accelerate huggingface-hub xgboost tsai fastai

echo.
echo Paso 4: Instalando PyTorch optimizado para NVIDIA (Serie 3000/4000)...
echo Descargando binarios CUDA 12.1+ (Esto tomara varios minutos...)
%PIP_CMD% torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Paso 5: Descargando pesos de los modelos IA (TimesFM y TSPulse)...
echo Esto descargara varios Gigabytes de redes neuronales desde HuggingFace.
echo Por favor ten paciencia, depende de tu conexion a internet...
.venv\Scripts\python -c "from huggingface_hub import snapshot_download; print('Descargando Google TimesFM...'); snapshot_download(repo_id='google/timesfm-2.0-500m-pytorch'); print('Descargando IBM TSPulse...'); snapshot_download(repo_id='ibm-granite/granite-timeseries-patchtsmixer')"

echo.
echo ========================================================
echo INSTALACION COMPLETADA EXITOSAMENTE
echo ========================================================
echo.
echo Para arrancar tu actualizacion diaria de IA:
echo   actualizar_ia.bat
echo.
echo Para arrancar el Dashboard web:
echo   .venv\Scripts\python server.py
echo.
pause
