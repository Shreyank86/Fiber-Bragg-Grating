# PINN FBG Sensor Dashboard

A beautiful, presentation-ready dashboard for visualizing Physics-Informed Neural Network (PINN) results for Fiber Bragg Grating (FBG) sensor analysis.

## Features

- **Metrics Overview**: Key performance indicators (MAE, RMSE, R², etc.)
- **Time Series Visualization**: Actual vs predicted wavelength shifts
- **Residual Analysis**: Error distribution and statistics
- **Model Comparison**: Linear baseline vs PINN performance
- **Bridge Simulation**: Real-world structural health monitoring visualization

## Installation

```bash
cd frontend
npm install
```

## Development

```bash
npm run dev
```

The dashboard will open at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Tech Stack

- **React 18** with TypeScript
- **Material-UI (MUI)** for beautiful, accessible components
- **Recharts** for interactive data visualization
- **Vite** for fast development and building

## Design Philosophy

- Physics-inspired color palette (deep blues, teals)
- Clean, readable typography (Inter/Roboto)
- Professional presentation-ready interface
- Desktop-optimized layout

