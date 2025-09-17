// For use in browser with script tags
(function() {
  const React = window.React;
  
  /**
   * GooeyText component for creating a morphing text effect with gooey transitions
   */
  function GooeyText({
    texts,
    morphTime = 1,
    cooldownTime = 0.25,
    className,
    textClassName
  }) {
    // Create refs for the two text elements that will morph between each other
    const text1Ref = React.useRef(null);
    const text2Ref = React.useRef(null);
  
    React.useEffect(() => {
      let textIndex = texts.length - 1;
      let time = new Date();
      let morph = 0;
      let cooldown = cooldownTime;
  
      // Set the morphing effect based on the fraction
      const setMorph = (fraction) => {
        if (text1Ref.current && text2Ref.current) {
          // Apply blur and opacity to the second text
          text2Ref.current.style.filter = `blur(${Math.min(8 / fraction - 8, 100)}px)`;
          text2Ref.current.style.opacity = `${Math.pow(fraction, 0.4) * 100}%`;
  
          // Apply inverse effect to the first text
          fraction = 1 - fraction;
          text1Ref.current.style.filter = `blur(${Math.min(8 / fraction - 8, 100)}px)`;
          text1Ref.current.style.opacity = `${Math.pow(fraction, 0.4) * 100}%`;
        }
      };
  
      // Reset the morphing effect during cooldown
      const doCooldown = () => {
        morph = 0;
        if (text1Ref.current && text2Ref.current) {
          text2Ref.current.style.filter = "";
          text2Ref.current.style.opacity = "100%";
          text1Ref.current.style.filter = "";
          text1Ref.current.style.opacity = "0%";
        }
      };
  
      // Apply the morphing effect
      const doMorph = () => {
        morph -= cooldown;
        cooldown = 0;
        let fraction = morph / morphTime;
  
        if (fraction > 1) {
          cooldown = cooldownTime;
          fraction = 1;
        }
  
        setMorph(fraction);
      };
  
      // Animation loop
      function animate() {
        requestAnimationFrame(animate);
        const newTime = new Date();
        const shouldIncrementIndex = cooldown > 0;
        const dt = (newTime.getTime() - time.getTime()) / 1000;
        time = newTime;
  
        cooldown -= dt;
  
        if (cooldown <= 0) {
          if (shouldIncrementIndex) {
            textIndex = (textIndex + 1) % texts.length;
            if (text1Ref.current && text2Ref.current) {
              text1Ref.current.textContent = texts[textIndex % texts.length];
              text2Ref.current.textContent = texts[(textIndex + 1) % texts.length];
            }
          }
          doMorph();
        } else {
          doCooldown();
        }
      }
  
      // Start the animation
      animate();
  
      // Clean up function (if needed)
      return () => {
        // Cleanup logic if needed
      };
    }, [texts, morphTime, cooldownTime]);
  
    return React.createElement(
      "div", 
      { className: `relative ${className || ''}` },
      [
        React.createElement(
          "svg", 
          { className: "absolute h-0 w-0", "aria-hidden": "true", focusable: "false", key: "svg" },
          React.createElement(
            "defs", 
            null,
            React.createElement(
              "filter", 
              { id: "threshold" },
              React.createElement(
                "feColorMatrix",
                {
                  in: "SourceGraphic",
                  type: "matrix",
                  values: "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 255 -140"
                }
              )
            )
          )
        ),
        React.createElement(
          "div",
          {
            className: "flex items-center justify-center",
            style: { filter: "url(#threshold)" },
            key: "container"
          },
          [
            React.createElement(
              "span",
              {
                ref: text1Ref,
                className: `absolute inline-block select-none text-center text-6xl md:text-[60pt] text-foreground ${textClassName || ''}`,
                key: "text1"
              }
            ),
            React.createElement(
              "span",
              {
                ref: text2Ref,
                className: `absolute inline-block select-none text-center text-6xl md:text-[60pt] text-foreground ${textClassName || ''}`,
                key: "text2"
              }
            )
          ]
        )
      ]
    );
  }
  
  // Export for browser/global use
  window.GooeyText = GooeyText;
})();