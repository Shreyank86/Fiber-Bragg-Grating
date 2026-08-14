# Quick Start Guide

## Step 1: Install Node.js (Required First!)

**Node.js is NOT installed on your system yet.**

1. **Download Node.js:**
   - Go to: https://nodejs.org/
   - Download the **LTS version** (recommended, v20.x.x or v18.x.x)
   - Choose the Windows Installer (.msi)

2. **Install Node.js:**
   - Run the downloaded installer
   - Click "Next" through the installation
   - Make sure "Add to PATH" is checked (should be by default)
   - Click "Install"

3. **Verify Installation:**
   - Open a NEW Command Prompt (close and reopen if you had one open)
   - Type: `node --version`
   - Should show: `v20.x.x` or `v18.x.x`
   - Type: `npm --version`
   - Should show: `10.x.x` or `9.x.x`

## Step 2: Install Project Dependencies

Once Node.js is installed:

1. **Open Command Prompt** and navigate to the frontend folder:
   ```bash
   cd C:\Users\anvit\Desktop\COEInternship\frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```
   
   This will read `package.json` and install all required packages.
   Wait for it to finish (may take 2-5 minutes).

## Step 3: Run the Dashboard

After installation completes:

```bash
npm run dev
```

The dashboard will automatically open in your browser at `http://localhost:3000`

## Troubleshooting

### If `npm install` fails:
- Make sure you're in the `frontend` folder
- Try: `npm cache clean --force` then `npm install` again
- Make sure you have internet connection

### If `npm run dev` doesn't work:
- Make sure `npm install` completed successfully
- Check that port 3000 is not already in use
- Try closing and reopening Command Prompt

## Summary

1. ✅ Install Node.js from nodejs.org
2. ✅ Open Command Prompt
3. ✅ `cd frontend`
4. ✅ `npm install`
5. ✅ `npm run dev`

That's it! The dashboard will open automatically.

