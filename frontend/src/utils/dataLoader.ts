import Papa from 'papaparse'

export interface DataPoint {
  Time: number
  CH1: number
  CH2: number
  CH3: number
  CH4: number
  Wavelength: number
  delta_lambda_pm?: number
}

export async function loadCSV(filePath: string): Promise<DataPoint[]> {
  try {
    const response = await fetch(filePath)
    const text = await response.text()
    
    return new Promise((resolve, reject) => {
      Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          const data = results.data.map((row: any) => ({
            Time: parseFloat(row.Time || row['Time (sec)'] || 0),
            CH1: parseFloat(row.CH1 || row['# CH 1'] || 0),
            CH2: parseFloat(row.CH2 || row['# CH 2'] || 0),
            CH3: parseFloat(row.CH3 || row['# CH 3'] || 0),
            CH4: parseFloat(row.CH4 || row['# CH 4'] || 0),
            Wavelength: parseFloat(row.Wavelength || 0),
            delta_lambda_pm: row.delta_lambda_pm ? parseFloat(row.delta_lambda_pm) : undefined,
          }))
          resolve(data)
        },
        error: (error) => reject(error),
      })
    })
  } catch (error) {
    console.error(`Error loading ${filePath}:`, error)
    return []
  }
}

// Mock data generator for demonstration (based on typical FBG sensor patterns)
export function generateMockData(length: number = 1000): DataPoint[] {
  const data: DataPoint[] = []
  const baseWavelength = 1524.0
  
  for (let i = 0; i < length; i++) {
    const time = i * 0.2
    const strainEffect = Math.sin(time / 100) * 0.5
    const tempEffect = Math.cos(time / 150) * 0.3
    const noise = (Math.random() - 0.5) * 0.1
    
    const wavelength = baseWavelength + strainEffect + tempEffect + noise
    const deltaLambda = (wavelength - baseWavelength) * 1000 // Convert to pm
    
    data.push({
      Time: time,
      CH1: 1,
      CH2: 0,
      CH3: 0,
      CH4: 0,
      Wavelength: wavelength,
      delta_lambda_pm: deltaLambda,
    })
  }
  
  return data
}

