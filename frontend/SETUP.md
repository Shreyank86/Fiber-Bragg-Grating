# Setup Instructions

## Quick Start

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open browser:**
   The dashboard will automatically open at `http://localhost:3000`

## Project Structure

```
frontend/
├── src/
│   ├── components/          # Dashboard components
│   │   ├── MetricsOverview.tsx
│   │   ├── TimeSeriesPlots.tsx
│   │   ├── ResidualAnalysis.tsx
│   │   ├── ModelComparison.tsx
│   │   └── BridgeSimulation.tsx
│   ├── utils/
│   │   └── dataLoader.ts     # CSV loading utilities
│   ├── App.tsx               # Main app component
│   ├── main.tsx              # Entry point
│   └── theme.ts              # Material-UI theme
├── public/                   # Static assets
├── package.json
└── vite.config.ts
```

## Loading Real Data

Currently, the dashboard uses mock data for demonstration. To load your actual CSV files:

1. Copy your CSV files to `frontend/public/data/`:
   - `STRAIN_CSV.csv`
   - `TEMP_CSV.csv`
   - `TEMP_STRAIN_CSV.csv`
   - `Week1_strain_final.csv`
   - `Week1_temp_final.csv`
   - `Week1_combined_final.csv`

2. Update the data loading in components to use `loadCSV()` from `utils/dataLoader.ts`

3. The `loadCSV()` function uses the Fetch API, so files must be served from the `public` folder or via a local server.

## API vs Local Access

**For a presentation/demo:** Local file access is perfectly fine! No API needed.

**If you want to add an API later:**
- Create a simple Express/Flask backend
- Serve CSV files or processed JSON data
- Update `dataLoader.ts` to fetch from API endpoints

For now, local access is recommended for simplicity and ease of deployment.

## Customization

### Colors
Edit `src/theme.ts` to change the color scheme. The current theme uses physics-inspired blues and teals.

### Metrics
Update the metrics values in `components/MetricsOverview.tsx` with your actual results.

### Charts
All charts use Recharts. Modify components in `src/components/` to customize visualizations.

## Building for Production

```bash
npm run build
```

Output will be in the `dist/` folder. You can deploy this to any static hosting service (Netlify, Vercel, GitHub Pages, etc.).

