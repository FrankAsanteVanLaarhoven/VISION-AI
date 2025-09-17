// navigation-menu.js - Shadcn UI Navigation Menu component adapted for vanilla JS
(function() {
  // Create the navigation menu trigger style
  const navigationMenuTriggerStyle = () => {
    return "group inline-flex h-9 w-max items-center justify-center rounded-md bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[active]:bg-accent/50 data-[state=open]:bg-accent/50";
  };

  // Create NavigationMenu component
  window.NavigationMenu = function NavigationMenu({ className, children, ...props }) {
    return React.createElement(
      "div",
      {
        className: window.cn(
          "relative z-10 flex max-w-max flex-1 items-center justify-center",
          className
        ),
        ...props
      },
      children,
      React.createElement(window.NavigationMenuViewport)
    );
  };

  // Create NavigationMenuList component
  window.NavigationMenuList = function NavigationMenuList({ className, children, ...props }) {
    return React.createElement(
      "div",
      {
        className: window.cn(
          "group flex flex-1 list-none items-center justify-center space-x-1",
          className
        ),
        ...props
      },
      children
    );
  };

  // Create NavigationMenuItem component
  window.NavigationMenuItem = function NavigationMenuItem(props) {
    return React.createElement("div", props);
  };

  // Create NavigationMenuTrigger component
  window.NavigationMenuTrigger = function NavigationMenuTrigger({ className, children, ...props }) {
    return React.createElement(
      "button",
      {
        className: window.cn(navigationMenuTriggerStyle(), "group", className),
        ...props
      },
      children,
      " ",
      React.createElement(
        "svg",
        {
          width: "12",
          height: "12",
          viewBox: "0 0 15 15",
          fill: "none",
          xmlns: "http://www.w3.org/2000/svg",
          className: "relative top-[1px] ml-1 h-3 w-3 transition duration-300 group-data-[state=open]:rotate-180"
        },
        React.createElement("path", {
          d: "M3.13523 6.15803C3.3241 5.95657 3.64052 5.94637 3.84197 6.13523L7.5 9.56464L11.158 6.13523C11.3595 5.94637 11.6759 5.95657 11.8648 6.15803C12.0536 6.35949 12.0434 6.67591 11.842 6.86477L7.84197 10.6148C7.64964 10.7951 7.35036 10.7951 7.15803 10.6148L3.15803 6.86477C2.95657 6.67591 2.94637 6.35949 3.13523 6.15803Z",
          fill: "currentColor",
          fillRule: "evenodd",
          clipRule: "evenodd"
        })
      )
    );
  };

  // Create NavigationMenuContent component
  window.NavigationMenuContent = function NavigationMenuContent({ className, children, ...props }) {
    return React.createElement(
      "div",
      {
        className: window.cn(
          "left-0 top-0 w-full data-[motion^=from-]:animate-in data-[motion^=to-]:animate-out data-[motion^=from-]:fade-in data-[motion^=to-]:fade-out data-[motion=from-end]:slide-in-from-right-52 data-[motion=from-start]:slide-in-from-left-52 data-[motion=to-end]:slide-out-to-right-52 data-[motion=to-start]:slide-out-to-left-52 md:absolute md:w-auto",
          className
        ),
        ...props
      },
      children
    );
  };

  // Create NavigationMenuLink component
  window.NavigationMenuLink = function NavigationMenuLink({ className, children, href, ...props }) {
    return React.createElement(
      "a",
      {
        href: href || "#",
        className,
        ...props
      },
      children
    );
  };

  // Create NavigationMenuViewport component
  window.NavigationMenuViewport = function NavigationMenuViewport({ className, ...props }) {
    return React.createElement(
      "div",
      {
        className: window.cn("absolute left-0 top-full flex justify-center")
      },
      React.createElement(
        "div",
        {
          className: window.cn(
            "origin-top-center relative mt-1.5 h-[var(--radix-navigation-menu-viewport-height)] w-full overflow-hidden rounded-md border bg-popover text-popover-foreground shadow data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-90 md:w-[var(--radix-navigation-menu-viewport-width)]",
            className
          ),
          ...props
        }
      )
    );
  };

  // Export navigation menu trigger style for use in other components
  window.navigationMenuTriggerStyle = navigationMenuTriggerStyle;
})();

