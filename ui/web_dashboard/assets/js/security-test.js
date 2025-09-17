// Security System Test Suite
// Run this to verify all protections are active

class SecurityTester {
    constructor() {
        this.tests = [];
        this.results = {};
        this.init();
    }

    init() {
        console.log('🧪 Security System Test Suite Starting...');
        
        // Wait for security system to be ready
        setTimeout(() => {
            this.runAllTests();
        }, 2000);
    }

    runAllTests() {
        console.log('🧪 Running Security Tests...');
        
        this.testCanvasProtection();
        this.testChartSecurity();
        this.testTamperingProtection();
        this.testRecoverySystem();
        
        this.displayResults();
    }

    testCanvasProtection() {
        console.log('🧪 Test 1: Canvas Protection');

        try {
            // Test if security system exists
            if (!window.chartSecurity) {
                this.addResult('Canvas Protection', 'FAIL', 'Security system not initialized');
                return;
            }

            // Create a test canvas
            const testCanvas = document.createElement('canvas');
            testCanvas.width = 2000; // Try to set large size
            testCanvas.height = 1500;

            // Check if security system caught it
            setTimeout(() => {
                if (testCanvas.width <= 800 && testCanvas.height <= 600) {
                    this.addResult('Canvas Protection', 'PASS', 'Canvas sizes properly limited');
                } else {
                    this.addResult('Canvas Protection', 'FAIL', `Canvas sizes not limited: ${testCanvas.width}x${testCanvas.height}`);
                }
            }, 100);

        } catch (error) {
            this.addResult('Canvas Protection', 'FAIL', 'Error testing canvas protection: ' + error.message);
        }
    }

    testChartSecurity() {
        console.log('🧪 Test 2: Chart Security');
        
        try {
            // Check if security system is active
            if (window.chartSecurity && window.chartSecurity.isInitialized) {
                this.addResult('Chart Security', 'PASS', 'Security system active');
                
                // Check security status
                const status = window.chartSecurity.getStatus();
                if (status.isActive) {
                    this.addResult('Security Status', 'PASS', `Protected charts: ${status.protectedCharts}`);
                } else {
                    this.addResult('Security Status', 'FAIL', 'Security system not active');
                }
            } else {
                this.addResult('Chart Security', 'FAIL', 'Security system not found');
            }
        } catch (error) {
            this.addResult('Chart Security', 'FAIL', 'Error: ' + error.message);
        }
    }

    testTamperingProtection() {
        console.log('🧪 Test 3: Tampering Protection');
        
        try {
            // Try to override security system
            const originalSecurity = window.chartSecurity;
            
            // Attempt to disable security
            try {
                window.chartSecurity = null;
                if (window.chartSecurity === originalSecurity) {
                    this.addResult('Tampering Protection', 'PASS', 'Security system protected from override');
                } else {
                    this.addResult('Tampering Protection', 'FAIL', 'Security system was overridden');
                }
            } catch (error) {
                this.addResult('Tampering Protection', 'PASS', 'Override attempt blocked: ' + error.message);
            }
        } catch (error) {
            this.addResult('Tampering Protection', 'FAIL', 'Error: ' + error.message);
        }
    }

    testRecoverySystem() {
        console.log('🧪 Test 4: Recovery System');
        
        try {
            // Check if error recovery is set up
            if (window.chartSecurity && window.chartSecurity.setupErrorRecovery) {
                this.addResult('Recovery System', 'PASS', 'Error recovery system active');
            } else {
                this.addResult('Recovery System', 'FAIL', 'Error recovery not found');
            }
        } catch (error) {
            this.addResult('Recovery System', 'FAIL', 'Error: ' + error.message);
        }
    }

    addResult(testName, status, message) {
        this.results[testName] = { status, message };
        console.log(`🧪 ${testName}: ${status} - ${message}`);
    }

    displayResults() {
        console.log('\n🧪 SECURITY TEST RESULTS:');
        console.log('========================');
        
        let passed = 0;
        let total = 0;
        
        Object.keys(this.results).forEach(testName => {
            const result = this.results[testName];
            const icon = result.status === 'PASS' ? '✅' : '❌';
            console.log(`${icon} ${testName}: ${result.message}`);
            
            if (result.status === 'PASS') passed++;
            total++;
        });
        
        console.log(`\n📊 Overall: ${passed}/${total} tests passed`);
        
        if (passed === total) {
            console.log('🎉 ALL SECURITY TESTS PASSED! System is fully protected.');
        } else {
            console.log('⚠️  Some tests failed. Security may be compromised.');
        }
        
        // Display security status in UI
        this.updateSecurityDisplay(passed, total);
    }

    updateSecurityDisplay(passed, total) {
        const securityStatus = document.querySelector('.security-status');
        if (securityStatus) {
            const indicator = securityStatus.querySelector('.security-indicator');
            const text = securityStatus.querySelector('.security-text');
            
            if (passed === total) {
                indicator.textContent = '🔒';
                indicator.className = 'security-indicator active';
                text.textContent = 'Secure';
                text.style.color = '#2ecc71';
            } else {
                indicator.textContent = '⚠️';
                indicator.className = 'security-indicator warning';
                text.textContent = 'Warning';
                text.style.color = '#f39c12';
            }
        }
    }
}

// Run security tests when page loads
document.addEventListener('DOMContentLoaded', () => {
    new SecurityTester();
});

// Also run tests after a delay to catch late initializations
setTimeout(() => {
    if (!window.securityTester) {
        window.securityTester = new SecurityTester();
    }
}, 3000);
