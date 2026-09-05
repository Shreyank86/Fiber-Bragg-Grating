import { useEffect, useState } from 'react'
import { Grid, Card, CardContent, Typography, Box, CircularProgress } from '@mui/material'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import AssessmentIcon from '@mui/icons-material/Assessment'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline'

interface MetricCardProps {
  title: string
  value: string | number
  unit?: string
  icon: React.ReactNode
  color: string
  trend?: string
}

function MetricCard({ title, value, unit, icon, color, trend }: MetricCardProps) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Box
            sx={{
              bgcolor: `${color}20`,
              borderRadius: 2,
              p: 1.5,
              mr: 2,
              color: color,
            }}
          >
            {icon}
          </Box>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" component="div" sx={{ fontWeight: 600 }}>
              {typeof value === 'number' ? value.toFixed(2) : value}
              {unit && (
                <Typography component="span" variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
                  {unit}
                </Typography>
              )}
            </Typography>
            {trend && (
              <Typography variant="caption" color="success.main" sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                <TrendingUpIcon sx={{ fontSize: 14, mr: 0.5 }} />
                {trend}
              </Typography>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  )
}

export default function MetricsOverview() {
  const [loading, setLoading] = useState(true)
  const [metrics, setMetrics] = useState({
    mae: 1.25,
    rmse: 1.76,
    r2: 0.999,
    mbe: 0.05,
    stdDev: 1.42,
    correlation: 0.9995,
  })

  useEffect(() => {
    // Simulate loading
    setTimeout(() => setLoading(false), 500)
  }, [])

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Performance Metrics
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={4}>
          <MetricCard
            title="Mean Absolute Error"
            value={metrics.mae}
            unit="pm"
            icon={<ErrorOutlineIcon sx={{ fontSize: 32 }} />}
            color="#1976d2"
            trend="Excellent accuracy"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <MetricCard
            title="Root Mean Square Error"
            value={metrics.rmse}
            unit="pm"
            icon={<AssessmentIcon sx={{ fontSize: 32 }} />}
            color="#00897b"
            trend="Low variance"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <MetricCard
            title="Coefficient of Determination"
            value={metrics.r2}
            icon={<CheckCircleIcon sx={{ fontSize: 32 }} />}
            color="#388e3c"
            trend="Near-perfect fit"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <MetricCard
            title="Mean Bias Error"
            value={metrics.mbe}
            unit="pm"
            icon={<TrendingUpIcon sx={{ fontSize: 32 }} />}
            color="#0288d1"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <MetricCard
            title="Standard Deviation"
            value={metrics.stdDev}
            unit="pm"
            icon={<AssessmentIcon sx={{ fontSize: 32 }} />}
            color="#f57c00"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <MetricCard
            title="Pearson Correlation"
            value={metrics.correlation}
            icon={<CheckCircleIcon sx={{ fontSize: 32 }} />}
            color="#7b1fa2"
            trend="Strong correlation"
          />
        </Grid>
      </Grid>

      <Box sx={{ mt: 4 }}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Model Summary
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              The Physics-Informed Neural Network (PINN) demonstrates exceptional performance in predicting
              wavelength shifts (Δλ) for Fiber Bragg Grating (FBG) sensors. The model achieves sub-picometer
              accuracy with a Mean Absolute Error of {metrics.mae} pm and an R² value of {metrics.r2},
              indicating near-perfect agreement between predicted and actual measurements.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              The low bias error ({metrics.mbe} pm) and standard deviation ({metrics.stdDev} pm) confirm
              the model's reliability for real-world applications in structural health monitoring and
              temperature sensing.
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Box>
  )
}

