// Real-time Charts for QEP-VLA Dashboard

const API_BASE = 'http://localhost:8000';

class RealtimeCharts {
    constructor() {
        this.charts = new Map();
        this.updateInterval = null;
        this.init();
    }

    init() {
        // Wait for Chart to be available (loaded after security)
        const waitForChart = () => {
            if (window.Chart) {
                this.setupCharts();
                this.startRealTimeUpdates();
            } else {
                setTimeout(waitForChart, 50);
            }
        };
        waitForChart();
    }

    setupCharts() {
        // Initialize real-time performance monitoring
        this.setupPerformanceChart();
        this.setupResourceChart();
        this.setupAnalyticsChart();
    }

    setupPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;
        // Reuse existing chart if already created elsewhere
        const existing = Chart.getChart(ctx);
        const chart = existing || new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU Usage',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Memory Usage',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });

        this.charts.set('performance', chart);
    }

    setupResourceChart() {
        const ctx = document.getElementById('resourceChart');
        if (!ctx) return;
        const existing = Chart.getChart(ctx);
        const chart = existing || new Chart(ctx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'Memory', 'GPU', 'Network'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        this.charts.set('resource', chart);
    }

    setupAnalyticsChart() {
        const ctx = document.getElementById('analyticsChart');
        if (!ctx) return;
        const existing = Chart.getChart(ctx);
        const chart = existing || new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'System Performance',
                    data: [],
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 90,
                        max: 100
                    }
                }
            }
        });

        this.charts.set('analytics', chart);
    }

    startRealTimeUpdates() {
        this.updateInterval = setInterval(() => {
            this.updateCharts();
        }, 5000); // Update every 5 seconds
    }

    async updateCharts() {
        // Try to fetch real metrics; fallback to simulated values on failure
        const now = new Date();
        const timeLabel = now.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        let metrics;
        try {
            const res = await fetch(`${API_BASE}/api/v1/system/metrics`);
            if (res.ok) {
                metrics = await res.json();
            }
        } catch (_) {}

        // Update performance chart
        const perfChart = this.charts.get('performance');
        if (perfChart) {
            // Derive values from metrics if available
            const cpuUsage = metrics?.edge_inference?.cpu_usage_percent ?? (Math.random() * 30 + 20);
            const memUsage = metrics?.edge_inference?.memory_usage_percent ?? (Math.random() * 40 + 30);

            perfChart.data.labels.push(timeLabel);
            perfChart.data.datasets[0].data.push(cpuUsage);
            perfChart.data.datasets[1].data.push(memUsage);

            // Keep only last 20 data points
            if (perfChart.data.labels.length > 20) {
                perfChart.data.labels.shift();
                perfChart.data.datasets[0].data.shift();
                perfChart.data.datasets[1].data.shift();
            }

            // Use security system's safe update method
            if (window.chartSecurity && window.chartSecurity.safeChartUpdate) {
                if (!window.chartSecurity.safeChartUpdate(perfChart, 'update', 'none')) {
                    console.warn('Realtime Charts: Performance chart update failed, recreating...');
                    this.recreateChart('performance');
                }
            } else {
                // Fallback to basic error handling
                try {
                    perfChart.update('none');
                } catch (error) {
                    console.warn('Realtime Charts: Performance chart update failed:', error);
                    this.recreateChart('performance');
                }
            }
        }

        // Update resource chart
        const resChart = this.charts.get('resource');
        if (resChart) {
            resChart.data.datasets[0].data = [
                metrics?.edge_inference?.cpu_usage_percent ?? (Math.random() * 50 + 20),
                metrics?.edge_inference?.memory_usage_percent ?? (Math.random() * 60 + 20),
                metrics?.edge_inference?.gpu_usage_percent ?? (Math.random() * 40 + 10),
                metrics?.edge_inference?.network_usage_percent ?? (Math.random() * 30 + 10)
            ];
            // Use security system's safe update method
            if (window.chartSecurity && window.chartSecurity.safeChartUpdate) {
                if (!window.chartSecurity.safeChartUpdate(resChart, 'update', 'none')) {
                    console.warn('Realtime Charts: Resource chart update failed, recreating...');
                    this.recreateChart('resource');
                }
            } else {
                // Fallback to basic error handling
                try {
                    resChart.update('none');
                } catch (error) {
                    console.warn('Realtime Charts: Resource chart update failed:', error);
                    this.recreateChart('resource');
                }
            }
        }

        // Update analytics chart
        const analyticsChart = this.charts.get('analytics');
        if (analyticsChart) {
            try {
            const performance = metrics?.quantum_privacy_transform?.quantum_enhancement_factor
                ? Math.min(100, 90 + (metrics.quantum_privacy_transform.quantum_enhancement_factor * 4))
                : (Math.random() * 5 + 95);
            
            analyticsChart.data.labels.push(timeLabel);
            analyticsChart.data.datasets[0].data.push(performance);

            // Keep only last 20 data points
            if (analyticsChart.data.labels.length > 20) {
                analyticsChart.data.labels.shift();
                analyticsChart.data.datasets[0].data.shift();
            }

                // Use security system's safe update method
                if (window.chartSecurity && window.chartSecurity.safeChartUpdate) {
                    if (!window.chartSecurity.safeChartUpdate(analyticsChart, 'update', 'none')) {
                        console.warn('Realtime Charts: Analytics chart update failed, recreating...');
                        this.recreateChart('analytics');
                    }
                } else {
                    // Fallback to basic error handling
                    analyticsChart.update('none');
                }
            } catch (error) {
                console.warn('Realtime Charts: Analytics chart update failed:', error);
                this.recreateChart('analytics');
            }
        }
    }

    stop() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    recreateChart(chartType) {
        console.log(`Realtime Charts: Attempting to recreate ${chartType} chart`);

        try {
            // Remove existing chart
            const existingChart = this.charts.get(chartType);
            if (existingChart && typeof existingChart.destroy === 'function') {
                existingChart.destroy();
            }

            // Clear canvas
            const canvas = document.getElementById(`${chartType}Chart`);
            if (canvas) {
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            }

            // Recreate chart after a delay
            setTimeout(() => {
                this.setupCharts();
            }, 500);

        } catch (error) {
            console.error(`Realtime Charts: Failed to recreate ${chartType} chart:`, error);
        }
    }
}

// Initialize real-time charts when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RealtimeCharts();
});
