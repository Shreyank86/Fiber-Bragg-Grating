import { useState } from 'react'
import { Box, Container, Tabs, Tab, AppBar, Toolbar, Typography } from '@mui/material'
import ScienceIcon from '@mui/icons-material/Science'
import DashboardIcon from '@mui/icons-material/Dashboard'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import CompareArrowsIcon from '@mui/icons-material/CompareArrows'
import AccountTreeIcon from '@mui/icons-material/AccountTree'

import MetricsOverview from './components/MetricsOverview'
import TimeSeriesPlots from './components/TimeSeriesPlots'
import ResidualAnalysis from './components/ResidualAnalysis'
import ModelComparison from './components/ModelComparison'
import BridgeSimulation from './components/BridgeSimulation'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`dashboard-tabpanel-${index}`}
      aria-labelledby={`dashboard-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  )
}

function App() {
  const [value, setValue] = useState(0)

  const handleChange = (_event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue)
  }

  return (
    <Box sx={{ flexGrow: 1, minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" elevation={0} sx={{ bgcolor: 'primary.main' }}>
        <Toolbar>
          <ScienceIcon sx={{ mr: 2, fontSize: 32 }} />
          <Typography variant="h5" component="div" sx={{ flexGrow: 1, fontWeight: 600 }}>
            Physics-Informed Neural Network
          </Typography>
          <Typography variant="subtitle1" sx={{ opacity: 0.9 }}>
            FBG Sensor Analysis Dashboard
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs
            value={value}
            onChange={handleChange}
            aria-label="dashboard navigation tabs"
            sx={{
              '& .MuiTab-root': {
                textTransform: 'none',
                fontWeight: 500,
                fontSize: '1rem',
                minHeight: 64,
              },
            }}
          >
            <Tab icon={<DashboardIcon />} iconPosition="start" label="Overview" />
            <Tab icon={<ShowChartIcon />} iconPosition="start" label="Time Series" />
            <Tab icon={<CompareArrowsIcon />} iconPosition="start" label="Residuals" />
            <Tab icon={<AccountTreeIcon />} iconPosition="start" label="Model Comparison" />
            <Tab icon={<ScienceIcon />} iconPosition="start" label="Bridge Simulation" />
          </Tabs>
        </Box>

        <TabPanel value={value} index={0}>
          <MetricsOverview />
        </TabPanel>
        <TabPanel value={value} index={1}>
          <TimeSeriesPlots />
        </TabPanel>
        <TabPanel value={value} index={2}>
          <ResidualAnalysis />
        </TabPanel>
        <TabPanel value={value} index={3}>
          <ModelComparison />
        </TabPanel>
        <TabPanel value={value} index={4}>
          <BridgeSimulation />
        </TabPanel>
      </Container>
    </Box>
  )
}

export default App

