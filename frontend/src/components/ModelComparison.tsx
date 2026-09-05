import { Card, CardContent, Typography, Box, Grid } from '@mui/material'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts'

export default function ModelComparison() {
  const comparisonData = [
    { metric: 'MAE (pm)', Linear: 56.3, PINN: 1.25 },
    { metric: 'RMSE (pm)', Linear: 82.3, PINN: 1.76 },
    { metric: 'R²', Linear: 0.923, PINN: 0.999 },
  ]

  const radarData = [
    { subject: 'Accuracy', Linear: 92, PINN: 99 },
    { subject: 'Precision', Linear: 85, PINN: 98 },
    { subject: 'Robustness', Linear: 75, PINN: 95 },
    { subject: 'Physics', Linear: 60, PINN: 100 },
    { subject: 'Speed', Linear: 100, PINN: 85 },
  ]

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Model Comparison: Linear Baseline vs PINN
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Performance Metrics Comparison
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="metric" stroke="#666" />
                  <YAxis stroke="#666" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #ccc',
                      borderRadius: 8,
                    }}
                  />
                  <Legend />
                  <Bar dataKey="Linear" fill="#1976d2" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="PINN" fill="#00897b" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Model Capabilities Radar
              </Typography>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#e0e0e0" />
                  <PolarAngleAxis dataKey="subject" stroke="#666" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#666" />
                  <Radar
                    name="Linear Baseline"
                    dataKey="Linear"
                    stroke="#1976d2"
                    fill="#1976d2"
                    fillOpacity={0.6}
                  />
                  <Radar
                    name="PINN"
                    dataKey="PINN"
                    stroke="#00897b"
                    fill="#00897b"
                    fillOpacity={0.6}
                  />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                Key Advantages
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" color="primary" sx={{ fontWeight: 600, mt: 2 }}>
                  PINN Model:
                </Typography>
                <Typography variant="body2" color="text.secondary" component="ul" sx={{ pl: 2 }}>
                  <li>45x improvement in MAE (1.25 pm vs 56.3 pm)</li>
                  <li>47x improvement in RMSE (1.76 pm vs 82.3 pm)</li>
                  <li>Physics-informed architecture ensures physical consistency</li>
                  <li>Better generalization to unseen data</li>
                  <li>Handles coupled strain-temperature effects</li>
                </Typography>

                <Typography variant="subtitle2" color="primary" sx={{ fontWeight: 600, mt: 3 }}>
                  Linear Baseline:
                </Typography>
                <Typography variant="body2" color="text.secondary" component="ul" sx={{ pl: 2 }}>
                  <li>Simple and interpretable</li>
                  <li>Fast inference time</li>
                  <li>Limited accuracy for complex interactions</li>
                  <li>Requires separate calibration</li>
                </Typography>

                <Box sx={{ mt: 3, p: 2, bgcolor: 'success.light', borderRadius: 2, opacity: 0.1 }}>
                  <Typography variant="body2" color="success.dark" sx={{ fontWeight: 500 }}>
                    The PINN model demonstrates superior performance while maintaining physical
                    interpretability through its physics-informed architecture.
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

