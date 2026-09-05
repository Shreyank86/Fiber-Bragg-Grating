@echo off
echo ========================================
echo PINN Dashboard Installation Script
echo ========================================
echo.

echo Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    echo Minimum version required: v18.0.0
    pause
    exit /b 1
)

echo Node.js found!
node --version
echo.

echo Checking npm...
npm --version
echo.

echo Installing dependencies...
echo This may take a few minutes...
echo.

npm install

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo Try running: npm cache clean --force
    echo Then run this script again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To start the dashboard, run:
echo   npm run dev
echo.
echo The dashboard will open at http://localhost:3000
echo.
pause

