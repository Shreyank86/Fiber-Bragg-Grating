import { useState, useEffect } from 'react'
import { Card, CardContent, Typography, Box, Grid, Slider, Button, ButtonGroup } from '@mui/material'
import Plot from 'react-plotly.js'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import PauseIcon from '@mui/icons-material/Pause'
import RestartAltIcon from '@mui/icons-material/RestartAlt'

interface BridgeData {
  time: number
  strain: number
  temperature: number
  deltaLambda: number
}

export default function Bridge3DVisualization() {
  const [timeStep, setTimeStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [bridgeData, setBridgeData] = useState<BridgeData[]>([])

  // Generate bridge simulation data
  useEffect(() => {
    const data: BridgeData[] = []
    const duration = 200 // seconds
    const dt = 0.2
    
    for (let t = 0; t < duration; t += dt) {
      // Simulate vehicle passing events
      const vehicle1 = t > 20 && t < 40 ? Math.sin((t - 20) * Math.PI / 20) * 50 : 0
      const vehicle2 = t > 80 && t < 100 ? Math.sin((t - 80) * Math.PI / 20) * 45 : 0
      const vehicle3 = t > 140 && t < 160 ? Math.sin((t - 140) * Math.PI / 20) * 55 : 0
      
      const strain = vehicle1 + vehicle2 + vehicle3 + (Math.random() - 0.5) * 2
      const temp = 20 + Math.sin(t / 50) * 5 + (Math.random() - 0.5) * 0.5
      const deltaLambda = strain * 1.2 + temp * 10 + (Math.random() - 0.5) * 0.5
      
      data.push({ time: t, strain, temperature: temp, deltaLambda })
    }
    
    setBridgeData(data)
  }, [])

  // Animation loop
  useEffect(() => {
    if (!isPlaying || bridgeData.length === 0) return

    const interval = setInterval(() => {
      setTimeStep((prev) => {
        const next = prev + 1
        return next >= bridgeData.length ? 0 : next
      })
    }, 100) // Update every 100ms

    return () => clearInterval(interval)
  }, [isPlaying, bridgeData.length])

  if (bridgeData.length === 0) {
    return <Box>Loading...</Box>
  }

  // Create 3D bridge surface data
  const bridgeLength = 50 // meters
  const bridgeWidth = 10 // meters
  const gridSize = 20

  const x = Array.from({ length: gridSize }, (_, i) => (i / (gridSize - 1)) * bridgeLength)
  const y = Array.from({ length: gridSize }, (_, i) => (i / (gridSize - 1)) * bridgeWidth - bridgeWidth / 2)

  // Current strain and temperature values
  const currentStrain = bridgeData[timeStep]?.strain || 0
  const currentTemp = bridgeData[timeStep]?.temperature || 20
  const currentDeltaLambda = bridgeData[timeStep]?.deltaLambda || 0

  // Create 3D surface: strain distribution across bridge
  const zStrain: number[][] = []
  const zTemp: number[][] = []
  const zCombined: number[][] = []

  for (let i = 0; i < gridSize; i++) {
    zStrain.push([])
    zTemp.push([])
    zCombined.push([])
    for (let j = 0; j < gridSize; j++) {
      // Strain distribution: higher near center, decays outward
      const centerX = bridgeLength / 2
      const centerY = 0
      const distFromCenter = Math.sqrt(
        Math.pow((x[i] - centerX) / bridgeLength, 2) + Math.pow((y[j] - centerY) / bridgeWidth, 2)
      )
      const strainValue = currentStrain * Math.exp(-distFromCenter * 3)
      
      // Temperature: uniform with slight gradient
      const tempValue = currentTemp + (y[j] / bridgeWidth) * 2
      
      // Combined effect
      const combinedValue = strainValue * 1.2 + tempValue * 10
      
      zStrain[i].push(strainValue)
      zTemp[i].push(tempValue)
      zCombined[i].push(combinedValue)
    }
  }

  const layout = {
    title: `Bridge Structural Health Monitoring - Time: ${currentDeltaLambda.toFixed(1)}s`,
    scene: {
      xaxis: { title: 'Bridge Length (m)' },
      yaxis: { title: 'Bridge Width (m)' },
      zaxis: { title: 'Strain (µε) / Δλ (pm)' },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.2 },
      },
      aspectmode: 'manual' as const,
      aspectratio: { x: 2, y: 1, z: 0.5 },
    },
    margin: { l: 0, r: 0, t: 50, b: 0 },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
  }

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        3D Bridge Structural Health Monitoring
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Strain Distribution Across Bridge Deck
                </Typography>
                <ButtonGroup>
                  <Button
                    variant={isPlaying ? 'contained' : 'outlined'}
                    onClick={() => setIsPlaying(!isPlaying)}
                    startIcon={isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                  >
                    {isPlaying ? 'Pause' : 'Play'}
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={() => {
                      setTimeStep(0)
                      setIsPlaying(false)
                    }}
                    startIcon={<RestartAltIcon />}
                  >
                    Reset
                  </Button>
                </ButtonGroup>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Time: {bridgeData[timeStep]?.time.toFixed(1)}s | 
                  Strain: {currentStrain.toFixed(1)} µε | 
                  Temperature: {currentTemp.toFixed(1)}°C | 
                  Δλ: {currentDeltaLambda.toFixed(1)} pm
                </Typography>
                <Slider
                  value={timeStep}
                  min={0}
                  max={bridgeData.length - 1}
                  step={1}
                  onChange={(_, value) => setTimeStep(value as number)}
                  sx={{ mt: 1 }}
                />
              </Box>

              <Plot
                data={[
                  {
                    type: 'surface',
                    x: x,
                    y: y,
                    z: zStrain,
                    colorscale: [
                      [0, '#1976d2'],
                      [0.5, '#42a5f5'],
                      [1, '#d32f2f'],
                    ],
                    colorbar: { title: 'Strain (µε)' },
                    name: 'Strain Distribution',
                  },
                ]}
                layout={layout}
                config={config}
                style={{ width: '100%', height: '600px' }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                Temperature Distribution
              </Typography>
              <Plot
                data={[
                  {
                    type: 'surface',
                    x: x,
                    y: y,
                    z: zTemp,
                    colorscale: [
                      [0, '#0288d1'],
                      [1, '#f57c00'],
                    ],
                    colorbar: { title: 'Temperature (°C)' },
                  },
                ]}
                layout={{
                  ...layout,
                  scene: {
                    ...layout.scene,
                    zaxis: { title: 'Temperature (°C)' },
                  },
                }}
                config={config}
                style={{ width: '100%', height: '400px' }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                Combined Wavelength Shift (Δλ)
              </Typography>
              <Plot
                data={[
                  {
                    type: 'surface',
                    x: x,
                    y: y,
                    z: zCombined,
                    colorscale: [
                      [0, '#00897b'],
                      [0.5, '#42a5f5'],
                      [1, '#d32f2f'],
                    ],
                    colorbar: { title: 'Δλ (pm)' },
                  },
                ]}
                layout={{
                  ...layout,
                  scene: {
                    ...layout.scene,
                    zaxis: { title: 'Δλ (pm)' },
                  },
                }}
                config={config}
                style={{ width: '100%', height: '400px' }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                Visualization Description
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                This 3D visualization shows the spatial distribution of strain, temperature, and wavelength shift
                across a bridge deck during structural health monitoring. The surface height represents the
                magnitude of each parameter, with color indicating intensity.
              </Typography>
              <Typography variant="body2" color="text.secondary" component="ul" sx={{ pl: 2 }}>
                <li>
                  <strong>Strain Distribution:</strong> Shows mechanical loading effects, with higher values
                  near the center where vehicles pass
                </li>
                <li>
                  <strong>Temperature Distribution:</strong> Shows thermal gradients across the bridge deck
                </li>
                <li>
                  <strong>Combined Δλ:</strong> The total wavelength shift measured by FBG sensors, combining
                  both strain and temperature effects
                </li>
                <li>
                  <strong>Interactive Controls:</strong> Use Play/Pause to animate through time, or drag the
                  slider to view specific moments. Rotate, zoom, and pan the 3D plots for detailed inspection.
                </li>
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

