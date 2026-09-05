# Troubleshooting 3D Bridge Visualization

## Issue: Can't see the 3D visualization

### Step 1: Install Dependencies
The 3D visualization requires Plotly.js. Make sure you've installed all dependencies:

```bash
cd frontend
npm install
```

This should install:
- `plotly.js` (^2.27.0)
- `react-plotly.js` (^2.6.0)
- `@types/plotly.js` (^2.12.29)

### Step 2: Check Browser Console
Open your browser's developer console (F12) and look for errors:
- If you see "Cannot find module 'react-plotly.js'", run `npm install` again
- If you see other errors, note them down

### Step 3: Navigate to the Right Tab
1. Click on the **"Bridge Simulation"** tab (5th tab in the main navigation)
2. You should see two sub-tabs: **"2D Time Series"** and **"3D Visualization"**
3. Click on **"3D Visualization"** to see the 3D plots

### Step 4: Verify Installation
Check if plotly is installed:
```bash
cd frontend
npm list plotly.js react-plotly.js
```

If they're not listed, install them:
```bash
npm install plotly.js react-plotly.js @types/plotly.js
```

### Step 5: Restart Dev Server
After installing dependencies, restart your development server:
1. Stop the current server (Ctrl+C)
2. Run `npm run dev` again
3. Refresh your browser

### Common Issues:

**Issue:** "Module not found: Can't resolve 'react-plotly.js'"
**Solution:** Run `npm install` in the frontend directory

**Issue:** 3D plots don't render (blank space)
**Solution:** 
- Check browser console for errors
- Make sure you're using a modern browser (Chrome, Firefox, Edge)
- Try clearing browser cache

**Issue:** Can't see the tabs
**Solution:** Make sure you're on the "Bridge Simulation" tab, then look for the sub-tabs below

### Alternative: Check if Component Loads
If the 3D component has errors, you might see the 2D visualization instead. Check the browser console for any React errors.

