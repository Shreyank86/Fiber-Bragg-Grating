# PINN FBG Sensor Dashboard - Project Summary

## Overview

A beautiful, presentation-ready web dashboard for visualizing Physics-Informed Neural Network (PINN) results for Fiber Bragg Grating (FBG) sensor analysis. Built specifically for project presentations with a focus on clarity, aesthetics, and scientific accuracy.

## Features Implemented

### 1. **Metrics Overview Dashboard**
- Key performance indicators displayed in beautiful metric cards
- Metrics include: MAE, RMSE, R², MBE, Standard Deviation, Correlation
- Color-coded icons and trend indicators
- Summary text explaining model performance

### 2. **Time Series Visualization**
- Main plot: Actual vs Predicted Δλ (wavelength shift)
- Separate components for Strain and Temperature effects
- Interactive tooltips and legends
- Clean, professional styling

### 3. **Residual Analysis**
- Residual plot showing prediction errors over time
- Histogram of residual distribution
- Statistical summary (MAE, RMSE, min/max)
- Zero-line reference for bias detection

### 4. **Model Comparison**
- Side-by-side bar chart comparing Linear Baseline vs PINN
- Radar chart showing model capabilities across multiple dimensions
- Detailed advantages list for each approach
- Visual performance improvements highlighted

### 5. **Bridge Simulation**
- Real-world structural health monitoring visualization
- Vehicle event detection in strain signals
- Temperature compensation visualization
- Multi-panel layout showing:
  - Measured wavelength shift
  - Strain decomposition (true vs PINN predicted)
  - Temperature decomposition (true vs PINN predicted)

## Design Philosophy

### Color Scheme
- **Primary**: Deep blue (#1976d2) - represents precision and science
- **Secondary**: Teal (#00897b) - represents innovation and technology
- **Accents**: Scientific color palette (blues, teals, oranges for different data types)
- **Background**: Clean white with subtle gray (#f5f7fa)

### Typography
- **Font**: Inter/Roboto - highly readable, modern sans-serif
- **Hierarchy**: Clear heading sizes (h1-h6) with appropriate weights
- **Spacing**: Generous line-height for readability

### UI Components
- Material-UI (MUI) for consistent, beautiful components
- Card-based layout with subtle shadows and hover effects
- Smooth transitions and animations
- Responsive grid system

## Technical Stack

- **React 18** - Modern React with hooks
- **TypeScript** - Type-safe development
- **Material-UI (MUI)** - Beautiful, accessible component library
- **Recharts** - Powerful, composable charting library
- **Vite** - Lightning-fast build tool
- **PapaParse** - CSV parsing utility (for future data integration)

## File Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── MetricsOverview.tsx      # Main metrics dashboard
│   │   ├── TimeSeriesPlots.tsx      # Time series visualizations
│   │   ├── ResidualAnalysis.tsx     # Error analysis
│   │   ├── ModelComparison.tsx      # Baseline vs PINN comparison
│   │   └── BridgeSimulation.tsx     # Bridge monitoring simulation
│   ├── utils/
│   │   └── dataLoader.ts            # CSV loading utilities
│   ├── App.tsx                      # Main app with navigation
│   ├── main.tsx                     # Entry point
│   └── theme.ts                     # Material-UI theme configuration
├── public/                          # Static assets
├── package.json                     # Dependencies
└── vite.config.ts                   # Vite configuration
```

## Current Status

✅ **Completed:**
- Project structure and setup
- All 5 main dashboard components
- Theme and styling
- Navigation system
- Mock data generation for demonstration

⏳ **Future Enhancements:**
- Integration with actual CSV data files
- Export functionality for charts
- Additional visualization options
- Real-time data updates (if needed)

## Usage for Presentation

1. **Start the dashboard:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. **Navigate through tabs:**
   - Overview: Start here for key metrics
   - Time Series: Show prediction accuracy
   - Residuals: Demonstrate error analysis
   - Model Comparison: Highlight PINN advantages
   - Bridge Simulation: Real-world application

3. **Presentation Tips:**
   - Use full-screen mode (F11)
   - Navigate tabs smoothly during presentation
   - Point out specific metrics and improvements
   - Use Bridge Simulation to show practical applications

## Customization

### Updating Metrics
Edit `src/components/MetricsOverview.tsx` and replace mock values with your actual results.

### Changing Colors
Modify `src/theme.ts` to adjust the color palette.

### Adding Data
1. Place CSV files in `public/data/`
2. Update components to use `loadCSV()` from `utils/dataLoader.ts`
3. Process data as needed for visualization

## Deployment

The dashboard can be deployed as a static site:

```bash
npm run build
```

Deploy the `dist/` folder to:
- Netlify
- Vercel
- GitHub Pages
- Any static hosting service

## Notes

- **No API Required**: Uses local file access for simplicity
- **Desktop Optimized**: Designed for presentation screens
- **Presentation Ready**: Clean, professional appearance
- **Extensible**: Easy to add new visualizations or features

