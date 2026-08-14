import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, Typography, Box, Grid, Slider, Button, ButtonGroup, Chip } from '@mui/material'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import PauseIcon from '@mui/icons-material/Pause'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import DirectionsCarIcon from '@mui/icons-material/DirectionsCar'
import ThermostatIcon from '@mui/icons-material/Thermostat'

export default function BridgeForcesVisualization() {
  const [timeStep, setTimeStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [currentStrain, setCurrentStrain] = useState(0)
  const [currentTemp, setCurrentTemp] = useState(20)
  const [vehiclePosition, setVehiclePosition] = useState<number | null>(null)
  const [timeOfDay, setTimeOfDay] = useState('Morning')

  // Simulate a day cycle (0-24 hours mapped to 0-200 seconds)
  const totalSteps = 200
  const dayDuration = 24 * 60 * 60 // 24 hours in seconds

  useEffect(() => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = 800
    canvas.height = 400

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw bridge image background (placeholder - you can replace with actual bridge image)
    drawBridge(ctx, canvas.width, canvas.height)

    // Calculate current time in day cycle
    const progress = timeStep / totalSteps
    const hours = progress * 24
    const currentHour = Math.floor(hours)

    // Determine time of day
    if (currentHour >= 6 && currentHour < 12) setTimeOfDay('Morning')
    else if (currentHour >= 12 && currentHour < 18) setTimeOfDay('Afternoon')
    else if (currentHour >= 18 && currentHour < 22) setTimeOfDay('Evening')
    else setTimeOfDay('Night')

    // Simulate temperature based on time of day (warmer midday)
    const baseTemp = 20
    const tempVariation = Math.sin((hours - 6) * Math.PI / 12) * 15 // Peak at noon
    const temp = baseTemp + tempVariation + (Math.random() - 0.5) * 2
    setCurrentTemp(Math.max(15, Math.min(35, temp)))

    // Simulate vehicle events
    const vehicleEvents = [
      { start: 20, duration: 20, intensity: 50 },
      { start: 80, duration: 20, intensity: 45 },
      { start: 140, duration: 20, intensity: 55 },
    ]

    let currentStrainValue = 0
    let vehiclePos: number | null = null

    for (const event of vehicleEvents) {
      if (timeStep >= event.start && timeStep < event.start + event.duration) {
        const eventProgress = (timeStep - event.start) / event.duration
        currentStrainValue = Math.sin(eventProgress * Math.PI) * event.intensity
        vehiclePos = eventProgress // 0 to 1 across bridge
        break
      }
    }

    setCurrentStrain(currentStrainValue)
    setVehiclePosition(vehiclePos)

    // Draw temperature gradient overlay
    drawTemperatureOverlay(ctx, canvas.width, canvas.height, temp, baseTemp)

    // Draw strain indicators (arrows showing deformation)
    if (currentStrainValue > 5) {
      drawStrainIndicators(ctx, canvas.width, canvas.height, currentStrainValue, vehiclePos)
    }

    // Draw vehicle if present
    if (vehiclePos !== null) {
      drawVehicle(ctx, canvas.width, canvas.height, vehiclePos)
    }

    // Draw sensor locations
    drawSensors(ctx, canvas.width, canvas.height)

    // Draw legend/info
    drawInfo(ctx, canvas.width, canvas.height, temp, currentStrainValue, currentHour)

  }, [timeStep])

  const drawBridge = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Draw bridge structure
    ctx.fillStyle = '#5a5a5a'
    ctx.fillRect(0, height * 0.6, width, height * 0.15) // Bridge deck

    // Draw supports
    const supportWidth = 30
    const supportHeight = height * 0.6
    ctx.fillRect(width * 0.2, height * 0.75, supportWidth, supportHeight)
    ctx.fillRect(width * 0.5, height * 0.75, supportWidth, supportHeight)
    ctx.fillRect(width * 0.8, height * 0.75, supportWidth, supportHeight)

    // Draw road markings
    ctx.strokeStyle = '#ffff00'
    ctx.lineWidth = 2
    ctx.setLineDash([10, 10])
    ctx.beginPath()
    ctx.moveTo(width * 0.1, height * 0.675)
    ctx.lineTo(width * 0.9, height * 0.675)
    ctx.stroke()
    ctx.setLineDash([])

    // Add text label
    ctx.fillStyle = '#333'
    ctx.font = '16px Arial'
    ctx.fillText('Bridge Deck', width * 0.5 - 40, height * 0.65)
  }

  const drawTemperatureOverlay = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    temp: number,
    baseTemp: number
  ) => {
    // Create temperature gradient (blue = cold, red = hot)
    const tempRange = 20 // ±20°C from base
    const normalizedTemp = (temp - baseTemp) / tempRange // -1 to 1
    const intensity = Math.abs(normalizedTemp)

    if (normalizedTemp > 0) {
      // Hot (red)
      ctx.fillStyle = `rgba(255, ${Math.floor(100 * (1 - intensity))}, 0, ${0.3 * intensity})`
    } else {
      // Cold (blue)
      ctx.fillStyle = `rgba(0, ${Math.floor(100 * (1 - intensity))}, 255, ${0.3 * intensity})`
    }

    ctx.fillRect(0, height * 0.6, width, height * 0.15)
  }

  const drawStrainIndicators = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    strain: number,
    vehiclePos: number | null
  ) => {
    if (vehiclePos === null) return

    const bridgeY = height * 0.675
    const vehicleX = width * 0.1 + vehiclePos * (width * 0.8)

    // Draw strain arrows (showing compression/deformation)
    ctx.strokeStyle = '#d32f2f'
    ctx.fillStyle = '#d32f2f'
    ctx.lineWidth = 2

    // Draw multiple arrows along the bridge showing strain distribution
    for (let i = 0; i < 5; i++) {
      const x = width * 0.1 + (i / 4) * (width * 0.8)
      const distanceFromVehicle = Math.abs(x - vehicleX) / width
      const localStrain = strain * Math.exp(-distanceFromVehicle * 5) // Decay with distance

      if (localStrain > 2) {
        // Draw arrow pointing down (compression)
        const arrowLength = localStrain * 2
        ctx.beginPath()
        ctx.moveTo(x, bridgeY)
        ctx.lineTo(x, bridgeY + arrowLength)
        ctx.lineTo(x - 5, bridgeY + arrowLength - 5)
        ctx.moveTo(x, bridgeY + arrowLength)
        ctx.lineTo(x + 5, bridgeY + arrowLength - 5)
        ctx.stroke()
      }
    }
  }

  const drawVehicle = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    position: number
  ) => {
    const x = width * 0.1 + position * (width * 0.8)
    const y = height * 0.65

    // Draw vehicle
    ctx.fillStyle = '#1976d2'
    ctx.fillRect(x - 15, y - 10, 30, 15)

    // Draw wheels
    ctx.fillStyle = '#000'
    ctx.beginPath()
    ctx.arc(x - 8, y + 5, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(x + 8, y + 5, 5, 0, Math.PI * 2)
    ctx.fill()
  }

  const drawSensors = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Draw FBG sensor locations
    const sensorY = height * 0.7
    const sensorPositions = [width * 0.2, width * 0.5, width * 0.8]

    ctx.fillStyle = '#00897b'
    sensorPositions.forEach((x) => {
      ctx.beginPath()
      ctx.arc(x, sensorY, 5, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.stroke()
    })
  }

  const drawInfo = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    temp: number,
    strain: number,
    hour: number
  ) => {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
    ctx.fillRect(10, 10, 250, 120)

    ctx.fillStyle = '#333'
    ctx.font = 'bold 14px Arial'
    ctx.fillText(`Time: ${hour}:00`, 20, 30)
    ctx.fillText(`Temperature: ${temp.toFixed(1)}°C`, 20, 50)
    ctx.fillText(`Strain: ${strain.toFixed(1)} µε`, 20, 70)
    ctx.fillText(`Δλ: ${(strain * 1.2 + temp * 10).toFixed(1)} pm`, 20, 90)

    // Legend
    ctx.font = '12px Arial'
    ctx.fillStyle = '#d32f2f'
    ctx.fillText('→ Strain (Vehicle Loading)', 20, 110)
    ctx.fillStyle = '#f57c00'
    ctx.fillText('→ Temperature Gradient', 20, 125)
  }

  // Animation loop
  useEffect(() => {
    if (!isPlaying) return

    const interval = setInterval(() => {
      setTimeStep((prev) => {
        const next = prev + 1
        return next >= totalSteps ? 0 : next
      })
    }, 200) // Update every 200ms

    return () => clearInterval(interval)
  }, [isPlaying])

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Bridge Forces Visualization
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Real-Time Bridge Monitoring
                  </Typography>
                  <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                    <Chip
                      icon={<ThermostatIcon />}
                      label={`${timeOfDay}: ${currentTemp.toFixed(1)}°C`}
                      color={currentTemp > 25 ? 'error' : currentTemp > 20 ? 'warning' : 'info'}
                    />
                    {vehiclePosition !== null && (
                      <Chip
                        icon={<DirectionsCarIcon />}
                        label={`Vehicle: ${currentStrain.toFixed(1)} µε`}
                        color="error"
                      />
                    )}
                  </Box>
                </Box>
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
                  Time: {Math.floor((timeStep / totalSteps) * 24)}:00 | 
                  Step: {timeStep}/{totalSteps}
                </Typography>
                <Slider
                  value={timeStep}
                  min={0}
                  max={totalSteps - 1}
                  step={1}
                  onChange={(_, value) => setTimeStep(value as number)}
                  sx={{ mt: 1 }}
                />
              </Box>

              <Box
                sx={{
                  border: '2px solid #e0e0e0',
                  borderRadius: 2,
                  overflow: 'hidden',
                  bgcolor: '#87CEEB', // Sky blue background
                }}
              >
                <canvas
                  ref={canvasRef}
                  style={{
                    width: '100%',
                    height: 'auto',
                    display: 'block',
                  }}
                />
              </Box>

              <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                <strong>Visualization Guide:</strong> Red arrows indicate strain from vehicle loading. 
                Color overlay shows temperature gradient (blue=cold, red=hot). Green circles mark FBG sensor locations.
                The simulation shows a full 24-hour cycle with vehicle events and temperature changes.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                Strain Distribution
              </Typography>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
                <Typography variant="body2" paragraph>
                  <strong>Current Strain:</strong> {currentStrain.toFixed(2)} µε
                </Typography>
                <Typography variant="body2" paragraph>
                  Strain increases when vehicles pass over the bridge. The red arrows show
                  the magnitude and distribution of mechanical loading.
                </Typography>
                <Typography variant="body2">
                  <strong>Vehicle Events:</strong> Three vehicle passing events occur during
                  the simulation, each causing a strain spike that decays as the vehicle moves away.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                Temperature Variation
              </Typography>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
                <Typography variant="body2" paragraph>
                  <strong>Current Temperature:</strong> {currentTemp.toFixed(1)}°C ({timeOfDay})
                </Typography>
                <Typography variant="body2" paragraph>
                  Temperature follows a daily cycle, peaking around midday (12:00-14:00)
                  and reaching minimum during night hours.
                </Typography>
                <Typography variant="body2">
                  <strong>Effect:</strong> Higher temperatures cause thermal expansion, contributing
                  to wavelength shift even without mechanical loading.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

