// Chart Security System - Permanent Protection Against Canvas Size Issues
// This system ensures charts never fail due to size problems

class ChartSecurityManager {
    constructor() {
        this.maxCanvasWidth = 800;
        this.maxCanvasHeight = 600;
        this.charts = new Map();
        this.protectedCanvases = new WeakSet();
        this.isInitialized = false;
        this.init();
    }

    init() {
        try {
            this.setupGlobalProtection();
            this.setupCanvasValidation();
            this.setupResizeProtection();
            this.setupErrorRecovery();
            this.isInitialized = true;
            console.log('🔒 Chart Security System: ACTIVE');

            // Force immediate protection
            this.protectAllCanvases();
        } catch (error) {
            console.error('🔒 Chart Security: Initialization failed:', error);
            this.isInitialized = false;
        }
    }

    setupGlobalProtection() {
        // Protect all existing canvases immediately
        this.protectAllCanvases();
        
        // Override Chart.js to add security layer (handles late loading)
        const installOverride = () => {
            if (typeof Chart === 'undefined' || window.__chartSecured) return false;

            // Disable Chart.js responsive features initially
            Chart.defaults.responsive = false;
            Chart.defaults.maintainAspectRatio = false;

            const originalChart = Chart;

            // Create secure Chart constructor
            const SecureChart = function(ctx, config) {
                try {
                    if (!ctx || !ctx.canvas) throw new Error('Invalid canvas context');
                    if (!chartSecurity.validateCanvas(ctx.canvas)) {
                        throw new Error('Canvas validation failed');
                    }
                    chartSecurity.applySecurityConstraints(ctx.canvas);
                    config = config || {};
                    config.options = config.options || {};
                    config.options.responsive = false;
                    config.options.maintainAspectRatio = false;
                    return new originalChart(ctx, config);
                } catch (error) {
                    console.error('Chart Security: Chart creation blocked:', error);
                    return chartSecurity.createFallbackChart(ctx, config);
                }
            };

            SecureChart.prototype = originalChart.prototype;
            SecureChart.Chart = originalChart.Chart;
            SecureChart.defaults = originalChart.defaults;
            SecureChart.registry = originalChart.registry;
            SecureChart.controllers = originalChart.controllers;
            SecureChart.elements = originalChart.elements;
            SecureChart.plugins = originalChart.plugins;
            SecureChart.scales = originalChart.scales;
            SecureChart.helpers = originalChart.helpers;

            window.Chart = SecureChart;
            window.__chartSecured = true;
            return true;
        };

        // Try now and also poll briefly for late Chart load
        if (!installOverride()) {
            let attempts = 0;
            const timer = setInterval(() => {
                attempts += 1;
                if (installOverride() || attempts > 40) {
                    clearInterval(timer);
                }
            }, 50);
        }
    }

    validateCanvas(canvas) {
        if (!canvas) return false;
        if (!canvas.getContext) return false;
        
        // Check if canvas size is within safe limits
        const maxSafeSize = 16384; // Maximum safe canvas size
        if (canvas.width > maxSafeSize || canvas.height > maxSafeSize) {
            console.warn('Chart Security: Canvas size exceeds safe limits, resizing...');
            this.resizeCanvas(canvas);
        }
        
        return true;
    }

    applySecurityConstraints(canvas) {
        if (!canvas || !(canvas instanceof HTMLCanvasElement)) return;
        if (this.protectedCanvases.has(canvas)) return; // already secured
        // Force canvas to safe dimensions
        const safeWidth = Math.min(canvas.width, this.maxCanvasWidth);
        const safeHeight = Math.min(canvas.height, this.maxCanvasHeight);
        
        if (canvas.width !== safeWidth || canvas.height !== safeHeight) {
            canvas.width = safeWidth;
            canvas.height = safeHeight;
            console.log('Chart Security: Canvas resized to safe dimensions:', safeWidth, 'x', safeHeight);
        }
        
        // Add CSS constraints
        canvas.style.maxWidth = '100%';
        canvas.style.maxHeight = '300px';
        canvas.style.width = 'auto';
        canvas.style.height = 'auto';
        
        // Prevent future size changes (define once, keep configurable in case of framework interactions)
        try {
            const widthDesc = Object.getOwnPropertyDescriptor(canvas, 'width');
            if (!widthDesc || widthDesc.configurable) {
                Object.defineProperty(canvas, 'width', {
                    get: function() { return safeWidth; },
                    set: function() { /* blocked */ },
                    configurable: true
                });
            }
        } catch (_) { /* ignore */ }

        try {
            const heightDesc = Object.getOwnPropertyDescriptor(canvas, 'height');
            if (!heightDesc || heightDesc.configurable) {
                Object.defineProperty(canvas, 'height', {
                    get: function() { return safeHeight; },
                    set: function() { /* blocked */ },
                    configurable: true
                });
            }
        } catch (_) { /* ignore */ }

        this.protectedCanvases.add(canvas);
    }

    resizeCanvas(canvas) {
        const aspectRatio = canvas.width / canvas.height;
        let newWidth = canvas.width;
        let newHeight = canvas.height;
        
        // Reduce size while maintaining aspect ratio
        if (newWidth > this.maxCanvasWidth) {
            newWidth = this.maxCanvasWidth;
            newHeight = newWidth / aspectRatio;
        }
        
        if (newHeight > this.maxCanvasHeight) {
            newHeight = this.maxCanvasHeight;
            newWidth = newHeight * aspectRatio;
        }
        
        canvas.width = Math.floor(newWidth);
        canvas.height = Math.floor(newHeight);
    }

    createFallbackChart(ctx, config) {
        console.log('🔒 Chart Security: Creating fallback chart');

        try {
            // Create a simple fallback visualization
            const canvas = ctx.canvas;
            const fallbackCtx = canvas.getContext('2d');

            // Clear canvas safely
            if (fallbackCtx) {
                fallbackCtx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw fallback message
                fallbackCtx.fillStyle = '#e74c3c';
                fallbackCtx.font = '16px Arial';
                fallbackCtx.textAlign = 'center';
                fallbackCtx.fillText('Chart Unavailable', canvas.width / 2, canvas.height / 2);
                fallbackCtx.fillText('Data Loading...', canvas.width / 2, canvas.height / 2 + 20);
            }
        } catch (error) {
            console.warn('🔒 Chart Security: Fallback chart creation failed:', error);
        }

        return {
            destroy: () => {},
            update: () => {},
            resize: () => {},
            isFallback: true
        };
    }

    setupCanvasValidation() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.setupCanvasValidation();
            });
            return;
        }
        
        // Monitor all canvas elements
        try {
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            this.validateAllCanvases(node);
                        }
                    });
                });
            });
            
            if (document.body) {
                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
                
                // Validate existing canvases
                this.validateAllCanvases(document.body);
            }
        } catch (error) {
            console.warn('Chart Security: MutationObserver setup failed, using fallback:', error);
            // Fallback: validate canvases periodically
            setInterval(() => {
                this.validateAllCanvases(document.body);
            }, 2000);
        }
    }

    validateAllCanvases(container) {
        const canvases = container.querySelectorAll ? container.querySelectorAll('canvas') : [];
        canvases.forEach(canvas => {
            this.validateCanvas(canvas);
            this.applySecurityConstraints(canvas);
        });
    }

    protectAllCanvases() {
        // Find and protect all canvases in the document
        const allCanvases = document.querySelectorAll('canvas');
        allCanvases.forEach(canvas => {
            this.validateCanvas(canvas);
            this.applySecurityConstraints(canvas);
        });
        
        // Also protect canvases added via innerHTML (wrap setter safely)
        if (!window.__innerHTMLSecured) {
            const desc = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML');
            if (desc && typeof desc.set === 'function' && typeof desc.get === 'function') {
                Object.defineProperty(Element.prototype, 'innerHTML', {
                    get: function() { return desc.get.call(this); },
                    set: function(value) {
                        if (!(this instanceof Element)) return desc.set.call(this, value);
                        const result = desc.set.call(this, value);
                        setTimeout(() => { try { chartSecurity.protectAllCanvases(); } catch(e) {} }, 0);
                        return result;
                    },
                    configurable: true,
                    enumerable: desc.enumerable
                });
                window.__innerHTMLSecured = true;
            }
        }
    }

    setupResizeProtection() {
        let resizeTimeout;
        
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleResize();
            }, 100);
        });
        
        // Prevent programmatic canvas size changes
        this.preventCanvasSizeChanges();
        
        // Override canvas getContext to add security
        this.overrideCanvasGetContext();
    }

    preventCanvasSizeChanges() {
        // Override canvas width/height setters
        const originalDescriptor = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, 'width');
        
        Object.defineProperty(HTMLCanvasElement.prototype, 'width', {
            set: function(value) {
                const maxWidth = 800;
                const safeValue = Math.min(value, maxWidth);
                if (value !== safeValue) {
                    console.warn('Chart Security: Canvas width limited to', maxWidth);
                }
                originalDescriptor.set.call(this, safeValue);
            },
            get: originalDescriptor.get,
            configurable: true
        });
        
        Object.defineProperty(HTMLCanvasElement.prototype, 'height', {
            set: function(value) {
                const maxHeight = 600;
                const safeValue = Math.min(value, maxHeight);
                if (value !== safeValue) {
                    console.warn('Chart Security: Canvas height limited to', maxHeight);
                }
                originalDescriptor.set.call(this, safeValue);
            },
            get: originalDescriptor.get,
            configurable: true
        });
    }

    overrideCanvasGetContext() {
        const originalGetContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(contextType, contextAttributes) {
            // Before getting context, ensure canvas is protected
            if (chartSecurity && chartSecurity.isInitialized) {
                chartSecurity.validateCanvas(this);
                chartSecurity.applySecurityConstraints(this);
            }
            
            return originalGetContext.call(this, contextType, contextAttributes);
        };
    }

    handleResize() {
        // Revalidate all canvases after resize
        this.validateAllCanvases(document.body);
        
        // Recreate charts if needed
        this.charts.forEach((chart, canvas) => {
            if (chart && typeof chart.resize === 'function') {
                try {
                    chart.resize();
                } catch (error) {
                    console.warn('Chart Security: Chart resize failed, recreating...');
                    this.recreateChart(canvas, chart);
                }
            }
        });
    }

    recreateChart(canvas, oldChart) {
        try {
            if (oldChart && typeof oldChart.destroy === 'function') {
                oldChart.destroy();
            }
            
            // Recreate chart with safe dimensions
            this.validateCanvas(canvas);
            this.applySecurityConstraints(canvas);
            
        } catch (error) {
            console.error('Chart Security: Chart recreation failed:', error);
        }
    }

    setupErrorRecovery() {
        // Global error handler for chart-related errors
        window.addEventListener('error', (event) => {
            if (event.error && event.error.message && 
                event.error.message.includes('Canvas')) {
                console.warn('Chart Security: Canvas error detected, attempting recovery...');
                this.recoverFromError(event);
            }
        });
        
        // Unhandled promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            if (event.reason && event.reason.message && 
                event.reason.message.includes('Canvas')) {
                console.warn('Chart Security: Canvas promise rejection detected, attempting recovery...');
                this.recoverFromError(event);
            }
        });
    }

    recoverFromError(event) {
        try {
            // Find and fix problematic canvases
            const canvases = document.querySelectorAll('canvas');
            canvases.forEach(canvas => {
                this.validateCanvas(canvas);
                this.applySecurityConstraints(canvas);
            });
            
            console.log('Chart Security: Error recovery completed');
        } catch (error) {
            console.error('Chart Security: Error recovery failed:', error);
        }
    }

    // Public API for manual chart registration
    registerChart(canvas, chart) {
        this.charts.set(canvas, chart);
        this.validateCanvas(canvas);
        this.applySecurityConstraints(canvas);
    }

    // Public API for manual chart cleanup
    unregisterChart(canvas) {
        this.charts.delete(canvas);
    }

    // Check if a chart is in error state
    isChartInError(chart) {
        if (!chart) return true;

        try {
            // Try to access chart properties to detect error state
            if (chart.canvas && chart.canvas.getContext) {
                const ctx = chart.canvas.getContext('2d');
                if (!ctx) return true;

                // Check if canvas is in error state by trying a simple operation
                const currentTransform = ctx.getTransform();
                return false; // Chart appears to be working
            }
        } catch (error) {
            console.warn('🔒 Chart Security: Chart detected in error state:', error);
            return true; // Chart is in error state
        }

        return true; // Default to error state if we can't check
    }

    // Safely update a chart with error handling
    safeChartUpdate(chart, updateMethod = 'update', ...args) {
        if (!chart) return false;

        if (this.isChartInError(chart)) {
            console.warn('🔒 Chart Security: Skipping update for chart in error state');
            return false;
        }

        try {
            if (typeof chart[updateMethod] === 'function') {
                chart[updateMethod](...args);
                return true;
            }
        } catch (error) {
            console.warn('🔒 Chart Security: Chart update failed:', error);
            return false;
        }

        return false;
    }

    // Get security status
    getStatus() {
        return {
            isActive: this.isInitialized,
            protectedCharts: this.charts.size,
            maxDimensions: {
                width: this.maxCanvasWidth,
                height: this.maxCanvasHeight
            }
        };
    }
}

// Initialize chart security system immediately
const chartSecurity = new ChartSecurityManager();

// Export for global access
window.chartSecurity = chartSecurity;

// Prevent tampering with the security system
Object.freeze(chartSecurity);
Object.freeze(ChartSecurityManager.prototype);

// Force immediate protection of all existing canvases
setTimeout(() => {
    chartSecurity.protectAllCanvases();
    console.log('🔒 Chart Security System: INITIALIZED AND ACTIVE');
}, 10);

console.log('🔒 Chart Security System: PERMANENTLY SECURED');
