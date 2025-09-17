// Privacy Controls for QEP-VLA Dashboard

const API_BASE = 'http://localhost:8000';

class PrivacyControls {
    constructor() {
        this.privacyLevel = 0.1; // Default epsilon value
        this.consentStatus = {};
        this.encryptionLevel = 'AES-256';
        this.init();
    }

    init() {
        this.setupPrivacyGauge();
        this.setupConsentManagement();
        this.setupEncryptionControls();
        this.startPrivacyMonitoring();
    }

    setupPrivacyGauge() {
        const canvas = document.getElementById('privacyGauge');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.drawPrivacyGauge(ctx, 99.8);
    }

    drawPrivacyGauge(ctx, percentage) {
        const centerX = ctx.canvas.width / 2;
        const centerY = ctx.canvas.height / 2;
        const radius = Math.min(centerX, centerY) - 20;

        // Clear canvas
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        // Draw gauge background
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = '#ecf0f1';
        ctx.lineWidth = 20;
        ctx.stroke();

        // Draw gauge fill
        const startAngle = -Math.PI / 2;
        const endAngle = startAngle + (percentage / 100) * 2 * Math.PI;
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        ctx.strokeStyle = this.getGaugeColor(percentage);
        ctx.stroke();

        // Draw center text
        ctx.fillStyle = '#2c3e50';
        ctx.font = 'bold 24px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${percentage}%`, centerX, centerY);

        // Draw gauge label
        ctx.font = '16px Arial';
        ctx.fillText('Privacy Compliance', centerX, centerY + 40);
    }

    getGaugeColor(percentage) {
        if (percentage >= 95) return '#2ecc71';
        if (percentage >= 80) return '#f39c12';
        return '#e74c3c';
    }

    setupConsentManagement() {
        // Create consent management interface
        const privacySection = document.querySelector('#privacy');
        if (!privacySection) return;

        const consentPanel = document.createElement('div');
        consentPanel.className = 'consent-panel';
        consentPanel.innerHTML = `
            <h4>Consent Management</h4>
            <div class="consent-items">
                <div class="consent-item">
                    <label class="consent-checkbox">
                        <input type="checkbox" checked onchange="privacyControls.updateConsent('data_collection', this.checked)">
                        <span class="checkmark"></span>
                        Data Collection
                    </label>
                    <span class="consent-status approved">Approved</span>
                </div>
                <div class="consent-item">
                    <label class="consent-checkbox">
                        <input type="checkbox" checked onchange="privacyControls.updateConsent('ai_processing', this.checked)">
                        <span class="checkmark"></span>
                        AI Processing
                    </label>
                    <span class="consent-status approved">Approved</span>
                </div>
                <div class="consent-item">
                    <label class="consent-checkbox">
                        <input type="checkbox" onchange="privacyControls.updateConsent('third_party', this.checked)">
                        <span class="checkmark"></span>
                        Third-party Sharing
                    </label>
                    <span class="consent-status pending">Pending</span>
                </div>
            </div>
        `;

        privacySection.appendChild(consentPanel);
    }

    setupEncryptionControls() {
        const privacySection = document.querySelector('#privacy');
        if (!privacySection) return;

        const encryptionPanel = document.createElement('div');
        encryptionPanel.className = 'encryption-panel';
        encryptionPanel.innerHTML = `
            <h4>Encryption & Security</h4>
            <div class="encryption-controls">
                <div class="control-group">
                    <label>Encryption Level:</label>
                    <select onchange="privacyControls.updateEncryption(this.value)">
                        <option value="AES-128">AES-128</option>
                        <option value="AES-256" selected>AES-256</option>
                        <option value="ChaCha20">ChaCha20</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Differential Privacy (ε):</label>
                    <input type="range" min="0.01" max="1.0" step="0.01" value="0.1" 
                           onchange="privacyControls.updatePrivacyLevel(this.value)">
                    <span class="epsilon-value">0.1</span>
                </div>
                <div class="control-group">
                    <label>Data Retention:</label>
                    <select onchange="privacyControls.updateRetention(this.value)">
                        <option value="7">7 days</option>
                        <option value="30" selected>30 days</option>
                        <option value="90">90 days</option>
                        <option value="365">1 year</option>
                    </select>
                </div>
            </div>
        `;

        privacySection.appendChild(encryptionPanel);
    }

    updateConsent(type, approved) {
        this.consentStatus[type] = approved;
        this.updatePrivacyMetrics();
        
        // Update UI
        const statusElement = document.querySelector(`[onchange*="${type}"]`).closest('.consent-item').querySelector('.consent-status');
        if (approved) {
            statusElement.className = 'consent-status approved';
            statusElement.textContent = 'Approved';
        } else {
            statusElement.className = 'consent-status denied';
            statusElement.textContent = 'Denied';
        }
    }

    async updateEncryption(level) {
        this.encryptionLevel = level;
        this.updatePrivacyMetrics();
        
        // Show notification
        this.showNotification(`Encryption updated to ${level}`, 'success');
        // No direct API config for encryption in backend example; placeholder
    }

    async updatePrivacyLevel(epsilon) {
        this.privacyLevel = parseFloat(epsilon);
        this.updatePrivacyMetrics();
        
        // Update epsilon display
        document.querySelector('.epsilon-value').textContent = epsilon;
        
        // Show notification
        this.showNotification(`Privacy level updated to ε=${epsilon}`, 'info');

        // Send to backend config
        try {
            await fetch(`${API_BASE}/api/v1/system/config`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ privacy_budget_epsilon: this.privacyLevel })
            });
        } catch (_) {}
    }

    async updateRetention(days) {
        this.updatePrivacyMetrics();
        this.showNotification(`Data retention set to ${days} days`, 'success');
        // Example: could be sent to backend if endpoint exists
    }

    updatePrivacyMetrics() {
        // Calculate privacy score based on current settings
        let score = 100;
        
        // Reduce score for higher epsilon values
        score -= (this.privacyLevel - 0.01) * 20;
        
        // Reduce score for denied consents
        Object.values(this.consentStatus).forEach(approved => {
            if (!approved) score -= 5;
        });
        
        // Ensure score doesn't go below 0
        score = Math.max(0, score);
        
        // Update gauge
        const canvas = document.getElementById('privacyGauge');
        if (canvas) {
            this.drawPrivacyGauge(canvas.getContext('2d'), score);
        }
        
        // Update metric bars
        this.updateMetricBars();
    }

    updateMetricBars() {
        // Update differential privacy bar
        const dpBar = document.querySelector('.privacy-metric:nth-child(1) .bar-fill');
        if (dpBar) {
            const dpScore = Math.max(0, 100 - (this.privacyLevel * 100));
            dpBar.style.width = `${dpScore}%`;
            dpBar.nextElementSibling.textContent = `${dpScore.toFixed(0)}%`;
        }
    }

    startPrivacyMonitoring() {
        // Monitor privacy compliance in real-time
        setInterval(() => {
            this.checkPrivacyCompliance();
        }, 10000); // Check every 10 seconds
    }

    checkPrivacyCompliance() {
        // Simulate privacy compliance checks
        const violations = [];
        
        if (this.privacyLevel > 0.5) {
            violations.push('High privacy risk (ε > 0.5)');
        }
        
        if (Object.values(this.consentStatus).some(status => !status)) {
            violations.push('Some consents not granted');
        }
        
        if (violations.length > 0) {
            this.showNotification(`Privacy violations detected: ${violations.join(', ')}`, 'warning');
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="this.parentElement.remove()">×</button>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    getPrivacyReport() {
        return {
            compliance: this.calculateComplianceScore(),
            settings: {
                epsilon: this.privacyLevel,
                encryption: this.encryptionLevel,
                consents: this.consentStatus
            },
            timestamp: new Date().toISOString()
        };
    }

    calculateComplianceScore() {
        let score = 100;
        score -= (this.privacyLevel - 0.01) * 20;
        
        Object.values(this.consentStatus).forEach(approved => {
            if (!approved) score -= 5;
        });
        
        return Math.max(0, score);
    }
}

// Initialize privacy controls when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.privacyControls = new PrivacyControls();
});
