// Quantum Loader for QEP-VLA Platform

class QuantumLoader {
    constructor() {
        this.isLoading = false;
        this.init();
    }

    init() {
        this.createLoader();
        this.setupEventListeners();
    }

    createLoader() {
        const loader = document.createElement('div');
        loader.id = 'quantum-loader';
        loader.className = 'quantum-loader';
        loader.innerHTML = `
            <div class="loader-content">
                <div class="quantum-spinner">
                    <div class="spinner-ring ring-1"></div>
                    <div class="spinner-ring ring-2"></div>
                    <div class="spinner-ring ring-3"></div>
                </div>
                <div class="loader-text">
                    <span class="loading-text">Initializing Quantum Systems</span>
                    <div class="loading-dots">
                        <span class="dot">.</span>
                        <span class="dot">.</span>
                        <span class="dot">.</span>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(loader);
    }

    show() {
        this.isLoading = true;
        const loader = document.getElementById('quantum-loader');
        if (loader) {
            loader.style.display = 'flex';
            this.animateDots();
        }
    }

    hide() {
        this.isLoading = false;
        const loader = document.getElementById('quantum-loader');
        if (loader) {
            loader.style.display = 'none';
        }
    }

    animateDots() {
        if (!this.isLoading) return;

        const dots = document.querySelectorAll('.loading-dots .dot');
        let currentDot = 0;

        const interval = setInterval(() => {
            dots.forEach((dot, index) => {
                dot.style.opacity = index === currentDot ? '1' : '0.3';
            });

            currentDot = (currentDot + 1) % dots.length;

            if (!this.isLoading) {
                clearInterval(interval);
            }
        }, 500);
    }

    setupEventListeners() {
        // Show loader on page load
        window.addEventListener('load', () => {
            setTimeout(() => {
                this.hide();
            }, 2000);
        });

        // Show loader on navigation
        document.addEventListener('click', (e) => {
            if (e.target.tagName === 'A' && e.target.href.includes('dashboard')) {
                this.show();
            }
        });
    }
}

// Initialize quantum loader when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new QuantumLoader();
});
