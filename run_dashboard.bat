@echo off
echo Iniciando servidor API local del Dashboard (con recalculador dinámico)...
start chrome "http://localhost:8000"
uv run --with yfinance --with pandas --with numpy python server.py
