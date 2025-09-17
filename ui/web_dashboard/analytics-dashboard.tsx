// components/AnalyticsDashboard.tsx
'use client'

import { useState, useEffect } from 'react'
import { Line, Bar, Pie } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
)

interface AnalyticsData {
  latencyHistory: number[]
  accuracyHistory: number[]
  commandTypes: { [key: string]: number }
  privacyMetrics: {
    epsilon: number
    dataMinimization: number
    encryptionCoverage: number
  }
  quantumMetrics: {
    enhancementFactor: number
    sensorReliability: number
    coherenceTime: number
  }
}

export function RealTimeAnalytics({ data }: { data: AnalyticsData }) {
  const [timeLabels, setTimeLabels] = useState<string[]>([])

  useEffect(() => {
    const generateTimeLabels = () => {
      const now = new Date()
      const labels = []
      for (let i = 19; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 30000) // 30-second intervals
        labels.push(time.toLocaleTimeString())
      }
      return labels
    }

    setTimeLabels(generateTimeLabels())
    
    const interval = setInterval(() => {
      setTimeLabels(generateTimeLabels())
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  const latencyChartData = {
    labels: timeLabels,
    datasets: [
      {
        label: 'Processing Latency (ms)',
        data: data.latencyHistory,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  }

  const accuracyChartData = {
    labels: timeLabels,
    datasets: [
      {
        label: 'Navigation Accuracy (%)',
        data: data.accuracyHistory,
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  }

  const commandDistributionData = {
    labels: Object.keys(data.commandTypes),
    datasets: [
      {
        data: Object.values(data.commandTypes),
        backgroundColor: [
          'rgba(59, 130, 246, 0.8)',
          'rgba(34, 197, 94, 0.8)',
          'rgba(251, 191, 36, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(168, 85, 247, 0.8)',
        ],
        borderColor: [
          'rgb(59, 130, 246)',
          'rgb(34, 197, 94)',
          'rgb(251, 191, 36)',
          'rgb(239, 68, 68)',
          'rgb(168, 85, 247)',
        ],
        borderWidth: 2,
      },
    ],
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
      {/* Latency Chart */}
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
        <h3 className="text-white font-semibold mb-4">Processing Latency</h3>
        <Line 
          data={latencyChartData} 
          options={{
            responsive: true,
            scales: {
              y: {
                beginAtZero: true,
                max: 100,
                grid: { color: 'rgba(75, 85, 99, 0.3)' },
                ticks: { color: 'rgba(156, 163, 175, 0.8)' }
              },
              x: {
                grid: { color: 'rgba(75, 85, 99, 0.3)' },
                ticks: { color: 'rgba(156, 163, 175, 0.8)' }
              }
            },
            plugins: {
              legend: { labels: { color: 'white' } }
            }
          }} 
        />
      </div>

      {/* Accuracy Chart */}
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
        <h3 className="text-white font-semibold mb-4">Navigation Accuracy</h3>
        <Line 
          data={accuracyChartData} 
          options={{
            responsive: true,
            scales: {
              y: {
                beginAtZero: false,
                min: 90,
                max: 100,
                grid: { color: 'rgba(75, 85, 99, 0.3)' },
                ticks: { color: 'rgba(156, 163, 175, 0.8)' }
              },
              x: {
                grid: { color: 'rgba(75, 85, 99, 0.3)' },
                ticks: { color: 'rgba(156, 163, 175, 0.8)' }
              }
            },
            plugins: {
              legend: { labels: { color: 'white' } }
            }
          }} 
        />
      </div>

      {/* Command Distribution */}
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
        <h3 className="text-white font-semibold mb-4">Command Distribution</h3>
        <Pie 
          data={commandDistributionData}
          options={{
            responsive: true,
            plugins: {
              legend: { 
                labels: { color: 'white' },
                position: 'bottom'
              }
            }
          }}
        />
      </div>

      {/* System Metrics */}
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
        <h3 className="text-white font-semibold mb-4">System Performance</h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-gray-300">Privacy Protection</span>
              <span className="text-cyan-400">ε={data.privacyMetrics.epsilon}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-green-400 h-2 rounded-full" 
                style={{ width: `${(1 - data.privacyMetrics.epsilon) * 100}%` }}
              ></div>
            </div>
          </div>
          
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-gray-300">Quantum Enhancement</span>
              <span className="text-purple-400">{data.quantumMetrics.enhancementFactor}x</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-purple-400 h-2 rounded-full" 
                style={{ width: `${(data.quantumMetrics.enhancementFactor / 3) * 100}%` }}
              ></div>
            </div>
          </div>
          
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-gray-300">Data Minimization</span>
              <span className="text-blue-400">{(data.privacyMetrics.dataMinimization * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-blue-400 h-2 rounded-full" 
                style={{ width: `${data.privacyMetrics.dataMinimization * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
