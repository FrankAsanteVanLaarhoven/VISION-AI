// header.js - Shadcn UI Header component adapted for vanilla JS
(function() {
  // Import Lucide icons
  const { Menu, MoveRight, X } = window.LucideIcons || {
    Menu: (props) => React.createElement("svg", { 
      xmlns: "http://www.w3.org/2000/svg", 
      width: "24", 
      height: "24", 
      viewBox: "0 0 24 24", 
      fill: "none", 
      stroke: "currentColor", 
      strokeWidth: "2", 
      strokeLinecap: "round", 
      strokeLinejoin: "round",
      ...props
    }, [
      React.createElement("line", { key: "1", x1: "4", x2: "20", y1: "12", y2: "12" }),
      React.createElement("line", { key: "2", x1: "4", x2: "20", y1: "6", y2: "6" }),
      React.createElement("line", { key: "3", x1: "4", x2: "20", y1: "18", y2: "18" })
    ]),
    MoveRight: (props) => React.createElement("svg", { 
      xmlns: "http://www.w3.org/2000/svg", 
      width: "24", 
      height: "24", 
      viewBox: "0 0 24 24", 
      fill: "none", 
      stroke: "currentColor", 
      strokeWidth: "2", 
      strokeLinecap: "round", 
      strokeLinejoin: "round",
      ...props
    }, [
      React.createElement("path", { key: "1", d: "M18 8L22 12L18 16" }),
      React.createElement("path", { key: "2", d: "M2 12H22" })
    ]),
    X: (props) => React.createElement("svg", { 
      xmlns: "http://www.w3.org/2000/svg", 
      width: "24", 
      height: "24", 
      viewBox: "0 0 24 24", 
      fill: "none", 
      stroke: "currentColor", 
      strokeWidth: "2", 
      strokeLinecap: "round", 
      strokeLinejoin: "round",
      ...props
    }, [
      React.createElement("path", { key: "1", d: "M18 6L6 18" }),
      React.createElement("path", { key: "2", d: "M6 6L18 18" })
    ])
  };

  // Create Header1 component
  window.Header1 = function Header1() {
    const [isOpen, setOpen] = React.useState(false);

    const navigationItems = [
      {
        title: "Home",
        href: "/",
        description: "",
      },
      {
        title: "Product",
        description: "Managing a small business today is already tough.",
        items: [
          {
            title: "Reports",
            href: "/reports",
          },
          {
            title: "Statistics",
            href: "/statistics",
          },
          {
            title: "Dashboards",
            href: "/dashboards",
          },
          {
            title: "Recordings",
            href: "/recordings",
          },
        ],
      },
      {
        title: "Company",
        description: "Managing a small business today is already tough.",
        items: [
          {
            title: "About us",
            href: "/about",
          },
          {
            title: "Fundraising",
            href: "/fundraising",
          },
          {
            title: "Investors",
            href: "/investors",
          },
          {
            title: "Contact us",
            href: "/contact",
          },
        ],
      },
    ];

    return React.createElement(
      "header",
      { className: "w-full z-40 fixed top-0 left-0 bg-background" },
      React.createElement(
        "div",
        { className: "container relative mx-auto min-h-20 flex gap-4 flex-row lg:grid lg:grid-cols-3 items-center" },
        React.createElement(
          "div",
          { className: "justify-start items-center gap-4 lg:flex hidden flex-row" },
          React.createElement(
            window.NavigationMenu,
            { className: "flex justify-start items-start" },
            React.createElement(
              window.NavigationMenuList,
              { className: "flex justify-start gap-4 flex-row" },
              navigationItems.map((item) =>
                React.createElement(
                  window.NavigationMenuItem,
                  { key: item.title },
                  item.href
                    ? React.createElement(
                        window.NavigationMenuLink,
                        null,
                        React.createElement(
                          window.Button,
                          { variant: "ghost" },
                          item.title
                        )
                      )
                    : [
                        React.createElement(
                          window.NavigationMenuTrigger,
                          { 
                            key: "trigger",
                            className: "font-medium text-sm",
                            onClick: () => {
                              // Toggle content visibility
                              const content = document.querySelector(`#content-${item.title}`);
                              if (content) {
                                content.style.display = content.style.display === "none" ? "block" : "none";
                              }
                            }
                          },
                          item.title
                        ),
                        React.createElement(
                          window.NavigationMenuContent,
                          { 
                            key: "content",
                            id: `content-${item.title}`,
                            className: "!w-[450px] p-4",
                            style: { display: "none", position: "absolute", top: "100%" }
                          },
                          React.createElement(
                            "div",
                            { className: "flex flex-col lg:grid grid-cols-2 gap-4" },
                            React.createElement(
                              "div",
                              { className: "flex flex-col h-full justify-between" },
                              React.createElement(
                                "div",
                                { className: "flex flex-col" },
                                React.createElement(
                                  "p",
                                  { className: "text-base" },
                                  item.title
                                ),
                                React.createElement(
                                  "p",
                                  { className: "text-muted-foreground text-sm" },
                                  item.description
                                )
                              ),
                              React.createElement(
                                window.Button,
                                { size: "sm", className: "mt-10" },
                                "Book a call today"
                              )
                            ),
                            React.createElement(
                              "div",
                              { className: "flex flex-col text-sm h-full justify-end" },
                              item.items?.map((subItem) =>
                                React.createElement(
                                  window.NavigationMenuLink,
                                  {
                                    href: subItem.href,
                                    key: subItem.title,
                                    className: "flex flex-row justify-between items-center hover:bg-muted py-2 px-4 rounded"
                                  },
                                  React.createElement("span", null, subItem.title),
                                  React.createElement(MoveRight, { className: "w-4 h-4 text-muted-foreground" })
                                )
                              )
                            )
                          )
                        )
                      ]
                )
              )
            )
          )
        ),
        React.createElement(
          "div",
          { className: "flex lg:justify-center" },
          React.createElement("p", { className: "font-semibold" }, "QEP-VLA")
        ),
        React.createElement(
          "div",
          { className: "flex justify-end w-full gap-4" },
          React.createElement(
            window.Button,
            { variant: "ghost", className: "hidden md:inline" },
            "Book a demo"
          ),
          React.createElement("div", { className: "border-r hidden md:inline" }),
          React.createElement(
            window.Button,
            { variant: "outline" },
            "Sign in"
          ),
          React.createElement(
            window.Button,
            null,
            "Get started"
          )
        ),
        React.createElement(
          "div",
          { className: "flex w-12 shrink lg:hidden items-end justify-end" },
          React.createElement(
            window.Button,
            { 
              variant: "ghost", 
              onClick: () => setOpen(!isOpen)
            },
            isOpen
              ? React.createElement(X, { className: "w-5 h-5" })
              : React.createElement(Menu, { className: "w-5 h-5" })
          ),
          isOpen &&
            React.createElement(
              "div",
              { className: "absolute top-20 border-t flex flex-col w-full right-0 bg-background shadow-lg py-4 container gap-8" },
              navigationItems.map((item) =>
                React.createElement(
                  "div",
                  { key: item.title },
                  React.createElement(
                    "div",
                    { className: "flex flex-col gap-2" },
                    item.href
                      ? React.createElement(
                          "a",
                          {
                            href: item.href,
                            className: "flex justify-between items-center"
                          },
                          React.createElement("span", { className: "text-lg" }, item.title),
                          React.createElement(MoveRight, { className: "w-4 h-4 stroke-1 text-muted-foreground" })
                        )
                      : [
                          React.createElement("p", { key: "title", className: "text-lg" }, item.title),
                          item.items &&
                            item.items.map((subItem) =>
                              React.createElement(
                                "a",
                                {
                                  key: subItem.title,
                                  href: subItem.href,
                                  className: "flex justify-between items-center"
                                },
                                React.createElement("span", { className: "text-muted-foreground" }, subItem.title),
                                React.createElement(MoveRight, { className: "w-4 h-4 stroke-1" })
                              )
                            )
                        ]
                  )
                )
              )
            )
        )
      )
    );
  };

  // Create HeaderDemo component
  window.HeaderDemo = function HeaderDemo() {
    return React.createElement(
      "div",
      { className: "block" },
      React.createElement(window.Header1)
    );
  };
})();

