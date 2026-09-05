# Installation Guide

## Prerequisites

1. **Node.js** (version 18 or higher)
   - Download from: https://nodejs.org/
   - Verify installation: `node --version` (should show v18.x.x or higher)
   - Verify npm: `npm --version` (should show 9.x.x or higher)

## Quick Install

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install all dependencies:**
   ```bash
   npm install
   ```
   
   This will read `package.json` and install all required packages automatically.

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   - The dashboard will automatically open at `http://localhost:3000`
   - Or manually navigate to that URL

## What Gets Installed

The `npm install` command will install:

- **React 18** - UI framework
- **Material-UI** - Component library
- **Recharts** - Charting library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **PapaParse** - CSV parser
- All other dependencies listed in `package.json`

## Troubleshooting

### If npm install fails:

1. **Clear npm cache:**
   ```bash
   npm cache clean --force
   ```

2. **Delete node_modules and package-lock.json:**
   ```bash
   rm -rf node_modules package-lock.json
   ```
   (On Windows: `rmdir /s node_modules` and delete `package-lock.json`)

3. **Try again:**
   ```bash
   npm install
   ```

### If port 3000 is already in use:

Edit `vite.config.ts` and change the port:
```typescript
server: {
  port: 3001,  // Change to any available port
  open: true
}
```

## Alternative: Using Yarn

If you prefer Yarn:
```bash
yarn install
yarn dev
```

## Alternative: Using pnpm

If you prefer pnpm:
```bash
pnpm install
pnpm dev
```

## Verify Installation

After installation, you should see:
- `node_modules/` folder created
- `package-lock.json` file created
- No error messages

Then run `npm run dev` and you should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

## Next Steps

Once the server is running:
1. Open `http://localhost:3000` in your browser
2. Navigate through the 5 dashboard tabs
3. Explore the visualizations
4. Let me know what improvements you'd like!

