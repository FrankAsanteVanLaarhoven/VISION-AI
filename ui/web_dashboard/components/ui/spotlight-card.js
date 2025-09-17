"use client";

/**
 * GlowCard/SpotlightCard Component
 * Creates a card with a dynamic glow effect that follows the cursor
 */
function GlowCard({ 
  children, 
  className = '', 
  glowColor = 'blue',
  size = 'md',
  width,
  height,
  customSize = false
}) {
  const glowColorMap = {
    blue: { base: 220, spread: 200 },
    purple: { base: 280, spread: 300 },
    green: { base: 120, spread: 200 },
    red: { base: 0, spread: 200 },
    orange: { base: 30, spread: 200 }
  };

  const sizeMap = {
    sm: 'w-48 h-64',
    md: 'w-64 h-80',
    lg: 'w-80 h-96'
  };

  // Effect is added via inline script in the HTML to avoid React dependencies
  const initScript = `
    (function() {
      const card = document.currentScript.parentElement;
      const inner = card.querySelector('[data-glow-inner]');
      
      function syncPointer(e) {
        const { clientX: x, clientY: y } = e;
        card.style.setProperty('--x', x.toFixed(2));
        card.style.setProperty('--xp', (x / window.innerWidth).toFixed(2));
        card.style.setProperty('--y', y.toFixed(2));
        card.style.setProperty('--yp', (y / window.innerHeight).toFixed(2));
      }
      
      document.addEventListener('pointermove', syncPointer);
    })();
  `;

  const { base, spread } = glowColorMap[glowColor];

  // Determine sizing
  const getSizeClasses = () => {
    if (customSize) {
      return ''; // Let className or inline styles handle sizing
    }
    return sizeMap[size];
  };

  const inlineStyles = {
    '--base': base,
    '--spread': spread,
    '--radius': '14',
    '--border': '3',
    '--backdrop': 'hsla(0, 0%, 60%, 0.12)',
    '--backup-border': 'var(--backdrop)',
    '--size': '200',
    '--outer': '1',
    '--border-size': 'calc(var(--border, 2) * 1px)',
    '--spotlight-size': 'calc(var(--size, 150) * 1px)',
    '--hue': 'calc(var(--base) + (var(--xp, 0) * var(--spread, 0)))',
    backgroundImage: `radial-gradient(
      var(--spotlight-size) var(--spotlight-size) at
      calc(var(--x, 0) * 1px)
      calc(var(--y, 0) * 1px),
      hsla(var(--hue, 210), calc(var(--saturation, 100) * 1%), calc(var(--lightness, 70) * 1%), var(--bg-spot-opacity, 0.1)), transparent
    )`,
    backgroundColor: 'var(--backdrop, transparent)',
    backgroundSize: 'calc(100% + (2 * var(--border-size))) calc(100% + (2 * var(--border-size)))',
    backgroundPosition: '50% 50%',
    backgroundAttachment: 'fixed',
    border: 'var(--border-size) solid var(--backup-border)',
    position: 'relative',
    touchAction: 'none',
    width: customSize && width ? (typeof width === 'number' ? `${width}px` : width) : undefined,
    height: customSize && height ? (typeof height === 'number' ? `${height}px` : height) : undefined
  };

  return `
    <div
      data-glow
      class="${getSizeClasses()} ${!customSize ? 'aspect-[3/4]' : ''} rounded-2xl relative grid grid-rows-[1fr_auto] shadow-[0_1rem_2rem_-1rem_black] p-4 gap-4 backdrop-blur-[5px] ${className}"
      style="${Object.entries(inlineStyles).map(([key, value]) => `${key}:${value}`).join(';')}"
    >
      <div data-glow-inner></div>
      <script>${initScript}</script>
      ${children || ''}
    </div>
  `;
}

// Export for use in JavaScript modules
export { GlowCard };
