// PERMANENT SECURITY CONFIGURATION
// This file ensures chart security settings can NEVER be changed

(function() {
    'use strict';
    
    // Lock down security settings
    const SECURITY_CONFIG = {
        MAX_CANVAS_WIDTH: 800,
        MAX_CANVAS_HEIGHT: 600,
        MAX_SAFE_SIZE: 16384,
        PROTECTION_LEVEL: 'MAXIMUM',
        AUTO_RECOVERY: true,
        PREVENT_TAMPERING: true
    };
    
    // Make config immutable
    Object.freeze(SECURITY_CONFIG);
    
    // Prevent any modifications to the security system
    const protectSecuritySystem = () => {
        if (window.chartSecurity) {
            // Freeze the security object
            Object.freeze(window.chartSecurity);
            
            // Prevent method overrides
            const methods = ['validateCanvas', 'applySecurityConstraints', 'protectAllCanvases'];
            methods.forEach(method => {
                if (window.chartSecurity[method]) {
                    Object.defineProperty(window.chartSecurity[method], 'prototype', {
                        writable: false,
                        configurable: false
                    });
                }
            });
            
            console.log('🔒 Security System: PERMANENTLY LOCKED');
        }
    };
    
    // Run protection when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', protectSecuritySystem);
    } else {
        protectSecuritySystem();
    }
    
    // Also protect after a delay to catch late initializations
    setTimeout(protectSecuritySystem, 1000);
    
    // Prevent any attempts to disable/override security while allowing reads
    (function secureGlobalRef(){
        const desc = Object.getOwnPropertyDescriptor(window, 'chartSecurity');
        // If already defined, freeze and seal the reference with a getter
        if (desc && ('value' in desc || typeof desc.get === 'function')) {
            const ref = desc.value || (desc.get && desc.get());
            if (ref) { try { Object.freeze(ref); } catch(_){} }
            Object.defineProperty(window, 'chartSecurity', {
                get: function(){ return ref; },
                set: function(){ console.warn('🔒 Security: Attempt to override chartSecurity blocked'); },
                configurable: false
            });
            return;
        }

        // Not defined yet; create a safe accessor that accepts the first set only
        let stored = undefined;
        Object.defineProperty(window, 'chartSecurity', {
            get: function(){ return stored; },
            set: function(value){
                if (stored === undefined && value) {
                    try { Object.freeze(value); } catch(_){}
                    stored = value;
                } else {
                    console.warn('🔒 Security: Attempt to override chartSecurity blocked');
                }
            },
            configurable: false
        });
    })();
    
    // Export config for reference
    window.SECURITY_CONFIG = SECURITY_CONFIG;
    
})();
