@echo off
rem Double-click to set up (first time only) and open the notebooks or the app.
rem Needs an internet connection the first time. Nothing is installed system-wide:
rem uv goes in your user folder, and Python plus the packages go in this folder (.venv).
setlocal
cd /d "%~dp0"
title Energy Demand Forecasting
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
rem Copy files instead of linking them, so the setup also works inside OneDrive or on another drive.
set "UV_LINK_MODE=copy"

where uv >nul 2>nul
if errorlevel 1 (
    echo Installing uv, the tool that sets up Python for this project. One time only...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    where uv >nul 2>nul
    if errorlevel 1 (
        echo.
        echo Could not install uv automatically. See START_HERE.md, section "If something goes wrong".
        pause
        exit /b 1
    )
)

echo Checking the project setup. The first time this downloads Python and the packages, about 2-5 minutes...
uv run --locked --no-dev python scripts\launch.py %*
if errorlevel 1 (
    echo.
    echo Something went wrong. See the messages above and START_HERE.md, section "If something goes wrong".
    pause
)
