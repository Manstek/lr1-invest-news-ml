@echo off
chcp 65001 >nul
rem Веб-интерфейс ИС: http://127.0.0.1:8000/
cd /d %~dp0webapp
call ..\.venv\Scripts\activate
python manage.py migrate
python manage.py runserver 8000
