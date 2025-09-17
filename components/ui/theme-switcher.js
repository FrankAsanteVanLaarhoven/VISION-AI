// Theme Switcher Component
(function() {
  const React = window.React;
  const cn = window.cn || function(...classes) {
    return classes.filter(Boolean).join(" ");
  };

  // Theme icons
  const SunIcon = function({ size = 20, className }) {
    return React.createElement(
      "svg",
      {
        xmlns: "http://www.w3.org/2000/svg",
        width: size,
        height: size,
        viewBox: "0 0 24 24",
        fill: "none",
        stroke: "currentColor",
        strokeWidth: 2,
        strokeLinecap: "round",
        strokeLinejoin: "round",
        className
      },
      [
        React.createElement("circle", { cx: "12", cy: "12", r: "5", key: "circle" }),
        React.createElement("line", { x1: "12", y1: "1", x2: "12", y2: "3", key: "line1" }),
        React.createElement("line", { x1: "12", y1: "21", x2: "12", y2: "23", key: "line2" }),
        React.createElement("line", { x1: "4.22", y1: "4.22", x2: "5.64", y2: "5.64", key: "line3" }),
        React.createElement("line", { x1: "18.36", y1: "18.36", x2: "19.78", y2: "19.78", key: "line4" }),
        React.createElement("line", { x1: "1", y1: "12", x2: "3", y2: "12", key: "line5" }),
        React.createElement("line", { x1: "21", y1: "12", x2: "23", y2: "12", key: "line6" }),
        React.createElement("line", { x1: "4.22", y1: "19.78", x2: "5.64", y2: "18.36", key: "line7" }),
        React.createElement("line", { x1: "18.36", y1: "5.64", x2: "19.78", y2: "4.22", key: "line8" })
      ]
    );
  };

  const MoonIcon = function({ size = 20, className }) {
    return React.createElement(
      "svg",
      {
        xmlns: "http://www.w3.org/2000/svg",
        width: size,
        height: size,
        viewBox: "0 0 24 24",
        fill: "none",
        stroke: "currentColor",
        strokeWidth: 2,
        strokeLinecap: "round",
        strokeLinejoin: "round",
        className
      },
      React.createElement("path", { d: "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" })
    );
  };

  const SystemIcon = function({ size = 20, className }) {
    return React.createElement(
      "svg",
      {
        xmlns: "http://www.w3.org/2000/svg",
        width: size,
        height: size,
        viewBox: "0 0 24 24",
        fill: "none",
        stroke: "currentColor",
        strokeWidth: 2,
        strokeLinecap: "round",
        strokeLinejoin: "round",
        className
      },
      [
        React.createElement("rect", { x: "2", y: "3", width: "20", height: "14", rx: "2", ry: "2", key: "rect" }),
        React.createElement("line", { x1: "8", y1: "21", x2: "16", y2: "21", key: "line1" }),
        React.createElement("line", { x1: "12", y1: "17", x2: "12", y2: "21", key: "line2" })
      ]
    );
  };

  // Settings icon
  const SettingsIcon = function({ size = 20, className }) {
    return React.createElement(
      "svg",
      {
        xmlns: "http://www.w3.org/2000/svg",
        width: size,
        height: size,
        viewBox: "0 0 24 24",
        fill: "none",
        stroke: "currentColor",
        strokeWidth: 2,
        strokeLinecap: "round",
        strokeLinejoin: "round",
        className
      },
      [
        React.createElement("path", { d: "M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z", key: "path1" }),
        React.createElement("circle", { cx: "12", cy: "12", r: "3", key: "circle" })
      ]
    );
  };

  // Theme Switcher Component
  function ThemeSwitcher() {
    const [theme, setTheme] = React.useState(() => {
      // Get initial theme from localStorage or default to system
      return localStorage.getItem('theme') || 'system';
    });
    
    const [isExpanded, setIsExpanded] = React.useState(false);

    // Apply theme on mount and when theme changes
    React.useEffect(() => {
      const applyTheme = (themeName) => {
        const root = document.documentElement;
        const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        
        console.log('Applying theme:', themeName, 'System preference:', systemTheme);
        
        // Remove all theme classes first
        root.classList.remove('light-theme', 'dark-theme');
        
        // Determine the actual theme to apply (resolving system preference if needed)
        const effectiveTheme = themeName === 'system' ? systemTheme : themeName;
        
        // Apply the appropriate theme class
        root.classList.add(`${effectiveTheme}-theme`);
        
        // Apply theme to body and other key elements
        if (effectiveTheme === 'light') {
          // Light theme
          document.body.style.background = '#ffffff';
          document.body.style.color = '#000000';
          
          // Find and update quantum particles background
          const particlesDiv = document.getElementById('quantum-particles');
          if (particlesDiv) {
            particlesDiv.style.background = '#ffffff';
          }
          
          // Find and update any sections with hardcoded backgrounds
          const sections = document.querySelectorAll('section');
          sections.forEach(section => {
            if (section.style.background === '#000000' || section.style.background === 'var(--background)') {
              section.style.background = '#ffffff';
            }
          });
          
          // Update any card backgrounds
          const cards = document.querySelectorAll('.card, .feature-card, .tech-card');
          cards.forEach(card => {
            card.style.background = 'rgba(240, 240, 240, 0.8)';
            card.style.color = '#000000';
          });
        } else {
          // Dark theme
          document.body.style.background = '#000000';
          document.body.style.color = '#ffffff';
          
          // Find and update quantum particles background
          const particlesDiv = document.getElementById('quantum-particles');
          if (particlesDiv) {
            particlesDiv.style.background = '#000000';
          }
          
          // Find and update any sections with hardcoded backgrounds
          const sections = document.querySelectorAll('section');
          sections.forEach(section => {
            if (section.style.background === '#ffffff' || section.style.background === 'var(--background)') {
              section.style.background = '#000000';
            }
          });
          
          // Update any card backgrounds
          const cards = document.querySelectorAll('.card, .feature-card, .tech-card');
          cards.forEach(card => {
            card.style.background = 'rgba(30, 30, 30, 0.8)';
            card.style.color = '#ffffff';
          });
        }
        
        // Save theme preference
        localStorage.setItem('theme', themeName);
      };
      
      // Apply the current theme
      applyTheme(theme);

      // Listen for system theme changes
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const handleChange = () => {
        if (theme === 'system') {
          const newSystemTheme = mediaQuery.matches ? 'dark' : 'light';
          applyTheme('system');
        }
      };

      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }, [theme]);

    const handleThemeChange = (newTheme) => {
      console.log('Theme changed to:', newTheme);
      setTheme(newTheme);
      setIsExpanded(false); // Collapse after selection
    };
    
    const toggleExpanded = () => {
      setIsExpanded(!isExpanded);
    };

    return React.createElement(
      "div",
      { 
        className: "theme-switcher",
        style: {
          position: "relative"
        }
      },
      [
        // Toggle button
        React.createElement(
          "button",
          {
            key: "toggle",
            onClick: toggleExpanded,
            className: "theme-toggle-button",
                   style: {
                     display: "flex",
                     alignItems: "center",
                     justifyContent: "center",
                     width: "100%",
                     height: "100%",
                     borderRadius: "50%",
                     cursor: "pointer",
                     transition: "all 0.3s ease",
                     zIndex: 1000,
                     position: "relative"
                   }
          },
          React.createElement(
            "div",
            {
              style: {
                position: "relative",
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }
            },
            [
                     // Current theme icon
                     theme === 'light'
                       ? React.createElement(SunIcon, { size: 22, key: "current-icon", style: { color: "#FFD700", filter: "drop-shadow(0 0 3px rgba(255, 255, 255, 0.8))" } })
                       : theme === 'dark'
                         ? React.createElement(MoonIcon, { size: 22, key: "current-icon", style: { color: "#E1E1E1", filter: "drop-shadow(0 0 3px rgba(255, 255, 255, 0.8))" } })
                         : React.createElement(SystemIcon, { size: 22, key: "current-icon", style: { color: "#7DF9FF", filter: "drop-shadow(0 0 3px rgba(255, 255, 255, 0.8))" } }),

                     // Settings gear icon with animation
                     React.createElement(SettingsIcon, {
                       size: 26,
                       key: "settings-icon",
                       style: {
                         position: "absolute",
                         opacity: 0.9,
                         color: "#FFFFFF",
                         filter: "drop-shadow(0 0 2px rgba(255, 255, 255, 0.8))",
                         animation: isExpanded ? "none" : "spin 4s linear infinite"
                       }
                     })
            ]
          )
        ),
        
        // Expanded theme selector
        isExpanded && React.createElement(
          "div",
          { 
            key: "expanded-menu",
            className: "theme-buttons",
            style: {
              position: "absolute",
              top: "50px",
              right: "0",
              display: "flex",
              flexDirection: "column",
              background: "rgba(0, 0, 0, 0.8)",
              borderRadius: "12px",
              padding: "10px",
              border: "1px solid rgba(255, 255, 255, 0.3)",
              boxShadow: "0 0 20px rgba(0, 0, 0, 0.5), 0 0 10px rgba(255, 255, 255, 0.2)",
              zIndex: 999,
              width: "180px",
              gap: "8px",
              backdropFilter: "blur(10px)",
              WebkitBackdropFilter: "blur(10px)"
            }
          },
          [
            // Title
            React.createElement(
              "div",
              {
                key: "title",
                style: {
                  color: "white",
                  fontSize: "14px",
                  fontWeight: "bold",
                  marginBottom: "8px",
                  textAlign: "center",
                  borderBottom: "1px solid rgba(255, 255, 255, 0.2)",
                  paddingBottom: "6px"
                }
              },
              "Select Theme"
            ),
            
            // Light theme option
            React.createElement(
              "button",
              {
                key: "light",
                onClick: () => handleThemeChange('light'),
                style: {
                  display: "flex",
                  alignItems: "center",
                  padding: "8px 12px",
                  borderRadius: "8px",
                  transition: "all 0.2s ease",
                  background: theme === 'light' ? "rgba(255, 255, 255, 0.9)" : "rgba(255, 255, 255, 0.1)",
                  color: theme === 'light' ? "#000000" : "#ffffff",
                  border: "none",
                  cursor: "pointer",
                  fontWeight: theme === 'light' ? "bold" : "normal",
                  boxShadow: theme === 'light' ? "0 0 10px rgba(255, 255, 255, 0.8)" : "none"
                }
              },
              [
                React.createElement(SunIcon, { size: 18, key: "light-icon", style: { marginRight: "10px" } }),
                "Light Mode"
              ]
            ),
            
            // Dark theme option
            React.createElement(
              "button",
              {
                key: "dark",
                onClick: () => handleThemeChange('dark'),
                style: {
                  display: "flex",
                  alignItems: "center",
                  padding: "8px 12px",
                  borderRadius: "8px",
                  transition: "all 0.2s ease",
                  background: theme === 'dark' ? "rgba(255, 255, 255, 0.9)" : "rgba(255, 255, 255, 0.1)",
                  color: theme === 'dark' ? "#000000" : "#ffffff",
                  border: "none",
                  cursor: "pointer",
                  fontWeight: theme === 'dark' ? "bold" : "normal",
                  boxShadow: theme === 'dark' ? "0 0 10px rgba(255, 255, 255, 0.8)" : "none"
                }
              },
              [
                React.createElement(MoonIcon, { size: 18, key: "dark-icon", style: { marginRight: "10px" } }),
                "Dark Mode"
              ]
            ),
            
            // System theme option
            React.createElement(
              "button",
              {
                key: "system",
                onClick: () => handleThemeChange('system'),
                style: {
                  display: "flex",
                  alignItems: "center",
                  padding: "8px 12px",
                  borderRadius: "8px",
                  transition: "all 0.2s ease",
                  background: theme === 'system' ? "rgba(255, 255, 255, 0.9)" : "rgba(255, 255, 255, 0.1)",
                  color: theme === 'system' ? "#000000" : "#ffffff",
                  border: "none",
                  cursor: "pointer",
                  fontWeight: theme === 'system' ? "bold" : "normal",
                  boxShadow: theme === 'system' ? "0 0 10px rgba(255, 255, 255, 0.8)" : "none"
                }
              },
              [
                React.createElement(SystemIcon, { size: 18, key: "system-icon", style: { marginRight: "10px" } }),
                "System Default"
              ]
            )
          ]
        )
      ]
    );
  }

  // Export for browser/global use
  window.ThemeSwitcher = ThemeSwitcher;
})();
