@echo off
title Actualizador de Modelos de IA
echo ========================================================
echo Actualizador de Modelos de Inteligencia Artificial
echo ========================================================
echo.
echo Este proceso tomara varios minutos. Asegurate de no cerrar la ventana.
echo.

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Detectado 'uv'.
    set PYTHON_CMD=uv run python
) else (
    echo [INFO] Detectado entorno estandar. Activando .venv...
    if exist ".venv\Scripts\activate.bat" (
        call .venv\Scripts\activate.bat
    )
    set PYTHON_CMD=python
)

echo [INFO] Configurando cache de datos en 1 hora para evitar re-descargas...
set YF_CACHE_SECONDS=3600

echo.
echo [1/6] Procesando Google TimesFM (Oraculo Principal)...
%PYTHON_CMD% models\precalculate_timesfm.py

echo.
echo [2/6] Procesando IBM TSPulse (Univariate)...
%PYTHON_CMD% models\finetune_tspulse.py

echo.
echo [3/6] Procesando IBM TSPulse (Multivariate)...
%PYTHON_CMD% models\finetune_tspulse_multi.py

echo.
echo [4/6] Procesando MiniRocket (Clasificacion Binaria)...
%PYTHON_CMD% models\train_minirocket.py

echo.
echo [5/6] Procesando MiniRocketPlus GPU (Probabilidades)...
%PYTHON_CMD% models\train_minirocket_gpu.py

echo.
echo [6/6] Procesando Ensamblado XGBoost Stack...
%PYTHON_CMD% models\train_xgboost_stack.py

echo.
echo ========================================================
echo ACTUALIZACION DE IA COMPLETADA EXITOSAMENTE.
echo Todas las predicciones .json han sido renovadas.
echo Ya puedes iniciar el Dashboard.
echo ========================================================
pause
