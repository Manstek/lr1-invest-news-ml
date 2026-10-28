@echo off
chcp 65001 >nul
rem Первичная установка: виртуальное окружение и зависимости
cd /d %~dp0
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
cd webapp
python manage.py migrate
echo.
echo Готово. Запустите run_ml_service.bat, затем run_webapp.bat,
echo после этого один раз: .venv\Scripts\python webapp\manage.py seed_demo
