@echo off
title Stop Dental AI Dashboard
echo Stopping Streamlit and Python screening processes...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /f /pid %%a 2>nul
)

taskkill /f /im streamlit.exe 2>nul
echo Dashboard stopped. Port 8501 freed.
pause
