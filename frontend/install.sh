#!/bin/bash

echo "========================================"
echo "PINN Dashboard Installation Script"
echo "========================================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed!"
    echo "Please install Node.js from https://nodejs.org/"
    echo "Minimum version required: v18.0.0"
    exit 1
fi

echo "Node.js found!"
node --version
echo ""

echo "Checking npm..."
npm --version
echo ""

echo "Installing dependencies..."
echo "This may take a few minutes..."
echo ""

npm install

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Installation failed!"
    echo "Try running: npm cache clean --force"
    echo "Then run this script again."
    exit 1
fi

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To start the dashboard, run:"
echo "  npm run dev"
echo ""
echo "The dashboard will open at http://localhost:3000"
echo ""

