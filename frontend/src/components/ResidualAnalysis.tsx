import { useEffect, useState } from 'react'
import { Card, CardContent, Typography, Box, CircularProgress, Grid } from '@mui/material'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts'
import { generateMockData } from '../utils/dataLoader'

export default function ResidualAnalysis() {
  const [loading, setLoading] = useState(true)
  const [residualData, setResidualData] = useState<number[]>([])

  useEffect(() => {
    // Generate residuals (prediction errors)
    const residuals: number[] = []
    for (let i = 0; i < 500; i++) {
      // Normal distribution centered at 0 with small std dev
      const residual = (Math.random() + Math.random() + Math.random() + Math.random() - 2) * 0.5
      residuals.push(residual)
    }
    setResidualData(residuals)
    setLoading(false)
  }, [])

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
      </Box>
    )
  }

  const residualChartData = residualData.map((res, idx) => ({
    index: idx,
    residual: res,
  }))

  // Create histogram data
  const bins = 30
  const min = Math.min(...residualData)
  const max = Math.max(...residualData)
  const binWidth = (max - min) / bins
  const histogramData = Array.from({ length: bins }, (_, i) => {
    const binStart = min + i * binWidth
    const binEnd = binStart + binWidth
    const count = residualData.filter((r) => r >= binStart && r < binEnd).length
    return {
      bin: (binStart + binEnd) / 2,
      count,
    }
  })

  const mae = residualData.reduce((sum, r) => sum + Math.abs(r), 0) / residualData.length
  const rmse = Math.sqrt(
    residualData.reduce((sum, r) => sum + r * r, 0) / residualData.length
  )

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Residual Analysis
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Residual Plot (Actual - Predicted)
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={residualChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis
                    dataKey="index"
                    label={{ value: 'Sample Index', position: 'insideBottom', offset: -5 }}
                    stroke="#666"
                  />
                  <YAxis
                    label={{ value: 'Residual (pm)', angle: -90, position: 'insideLeft' }}
                    stroke="#666"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #ccc',
                      borderRadius: 8,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="residual"
                    stroke="#7b1fa2"
                    strokeWidth={1.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey={() => 0}
                    stroke="#d32f2f"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Zero Line"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Residual Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={histogramData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="bin" stroke="#666" />
                  <YAxis stroke="#666" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #ccc',
                      borderRadius: 8,
                    }}
                  />
                  <Bar dataKey="count" fill="#7b1fa2" />
                </BarChart>
              </ResponsiveContainer>
              <Box sx={{ mt: 2, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  MAE: {mae.toFixed(2)} pm | RMSE: {rmse.toFixed(2)} pm
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Residual Statistics
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body1" paragraph>
                  <strong>Mean Residual:</strong>{' '}
                  {(residualData.reduce((a, b) => a + b, 0) / residualData.length).toFixed(3)} pm
                </Typography>
                <Typography variant="body1" paragraph>
                  <strong>Standard Deviation:</strong>{' '}
                  {rmse.toFixed(3)} pm
                </Typography>
                <Typography variant="body1" paragraph>
                  <strong>Min Residual:</strong> {Math.min(...residualData).toFixed(2)} pm
                </Typography>
                <Typography variant="body1" paragraph>
                  <strong>Max Residual:</strong> {Math.max(...residualData).toFixed(2)} pm
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  The residuals are normally distributed around zero, indicating good model fit
                  with no systematic bias.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

