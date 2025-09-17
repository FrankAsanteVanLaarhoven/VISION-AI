// Monitoring Section Test
// Tests that the monitoring section loads without canvas errors

class MonitoringTester {
    constructor() {
        this.init();
    }

    init() {
        // Wait for DOM and security system to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.runTests());
        } else {
            this.runTests();
        }
    }

    runTests() {
        console.log('🧪 Testing Monitoring Section...');

        setTimeout(() => {
            this.testMonitoringSection();
            this.testCanvasElements();
            this.testChartInitialization();
        }, 1000);
    }

    testMonitoringSection() {
        const monitoringSection = document.getElementById('monitoring');
        if (monitoringSection) {
            console.log('✅ Monitoring section found');
        } else {
            console.error('❌ Monitoring section not found');
        }
    }

    testCanvasElements() {
        const canvases = document.querySelectorAll('#monitoring canvas');
        console.log(`📊 Found ${canvases.length} canvases in monitoring section`);

        canvases.forEach((canvas, index) => {
            try {
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    console.log(`✅ Canvas ${index + 1}: ${canvas.width}x${canvas.height}`);
                } else {
                    console.error(`❌ Canvas ${index + 1}: No context available`);
                }
            } catch (error) {
                console.error(`❌ Canvas ${index + 1}: Error - ${error.message}`);
            }
        });
    }

    testChartInitialization() {
        // Test if charts can be created without errors
        const testCanvas = document.createElement('canvas');
        testCanvas.width = 400;
        testCanvas.height = 200;
        document.body.appendChild(testCanvas);

        try {
            if (window.Chart) {
                const testChart = new Chart(testCanvas.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: ['A', 'B', 'C'],
                        datasets: [{
                            data: [1, 2, 3]
                        }]
                    },
                    options: {
                        responsive: false,
                        maintainAspectRatio: false
                    }
                });
                console.log('✅ Chart creation successful');
                testChart.destroy();
            } else {
                console.log('⚠️ Chart.js not loaded yet');
            }
        } catch (error) {
            console.error('❌ Chart creation failed:', error.message);
        }

        // Clean up
        document.body.removeChild(testCanvas);
    }
}

// Initialize monitoring tests
new MonitoringTester();
