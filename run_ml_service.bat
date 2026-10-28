@echo off
chcp 65001 >nul
rem ML-сервис тональности: http://127.0.0.1:8001/docs
cd /d %~dp0
call .venv\Scripts\activate
uvicorn ml_service.main:app --port 8001
