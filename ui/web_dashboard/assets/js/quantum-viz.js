// Quantum Visualization for QEP-VLA Platform

class QuantumVisualizer {
    constructor() {
        this.particles = [];
        this.animationId = null;
        this.init();
    }

    init() {
        this.createParticles();
        this.animate();
        this.setupEventListeners();
    }

    createParticles() {
        const particleContainer = document.getElementById('quantum-particles');
        if (!particleContainer) return;

        for (let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.className = 'quantum-particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 2 + 's';
            particle.style.animationDuration = (Math.random() * 3 + 2) + 's';
            particleContainer.appendChild(particle);
        }
    }

    animate() {
        // Quantum particle animation
        const particles = document.querySelectorAll('.quantum-particle');
        particles.forEach(particle => {
            particle.style.transform = `translate(${Math.random() * 20 - 10}px, ${Math.random() * 20 - 10}px)`;
        });

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    setupEventListeners() {
        // Add interactive quantum effects
        document.addEventListener('mousemove', (e) => {
            this.createRipple(e.clientX, e.clientY);
        });
    }

    createRipple(x, y) {
        const ripple = document.createElement('div');
        ripple.className = 'quantum-ripple';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        document.body.appendChild(ripple);

        setTimeout(() => {
            ripple.remove();
        }, 1000);
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Initialize quantum visualizer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new QuantumVisualizer();
});
