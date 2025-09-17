// Tubelight Navbar Component
window.TubelightNavBar = function({ items, className = "" }) {
    const [activeTab, setActiveTab] = React.useState(items[0].name);
    const [isMobile, setIsMobile] = React.useState(false);

    React.useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth < 768);
        };

        handleResize();
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    return React.createElement("div", {
        className: window.cn(
            "fixed bottom-0 sm:top-0 left-1/2 -translate-x-1/2 z-50 mb-6 sm:pt-6",
            className
        )
    }, React.createElement("div", {
        className: "flex items-center gap-3 bg-background/5 border border-border backdrop-blur-lg py-1 px-1 rounded-full shadow-lg"
    }, items.map((item) => {
        const Icon = item.icon;
        const isActive = activeTab === item.name;

        return React.createElement("a", {
            key: item.name,
            href: item.url,
            onClick: (e) => {
                e.preventDefault();
                setActiveTab(item.name);
                // Handle navigation
                if (item.url && item.url !== '#') {
                    window.location.href = item.url;
                }
            },
            className: window.cn(
                "relative cursor-pointer text-sm font-semibold px-6 py-2 rounded-full transition-colors",
                "text-foreground/80 hover:text-primary",
                isActive && "bg-muted text-primary"
            )
        }, React.createElement("span", {
            className: "hidden md:inline"
        }, item.name), React.createElement("span", {
            className: "md:hidden"
        }, React.createElement(Icon, {
            size: 18,
            strokeWidth: 2.5
        })), isActive && React.createElement(window.Motion.div, {
            layoutId: "lamp",
            className: "absolute inset-0 w-full bg-primary/5 rounded-full -z-10",
            initial: false,
            transition: {
                type: "spring",
                stiffness: 300,
                damping: 30
            }
        }, React.createElement("div", {
            className: "absolute -top-2 left-1/2 -translate-x-1/2 w-8 h-1 bg-primary rounded-t-full"
        }, React.createElement("div", {
            className: "absolute w-12 h-6 bg-primary/20 rounded-full blur-md -top-2 -left-2"
        }), React.createElement("div", {
            className: "absolute w-8 h-6 bg-primary/20 rounded-full blur-md -top-1"
        }), React.createElement("div", {
            className: "absolute w-4 h-4 bg-primary/20 rounded-full blur-sm top-0 left-2"
        }))));
    })));
};

// NavBar Demo Component
window.TubelightNavBarDemo = function() {
    const navItems = [
        { 
            name: 'Platform', 
            url: '/enhanced-dashboard.html', 
            icon: window.LucideIcons?.Home || function() { return React.createElement("div", { className: "w-4 h-4" }); }
        },
        { 
            name: 'Resources', 
            url: '/ui/web_dashboard/command-center.html', 
            icon: window.LucideIcons?.FileText || function() { return React.createElement("div", { className: "w-4 h-4" }); }
        },
        { 
            name: 'Projects', 
            url: '/enhanced-dashboard.html', 
            icon: window.LucideIcons?.Briefcase || function() { return React.createElement("div", { className: "w-4 h-4" }); }
        },
        { 
            name: 'Features', 
            url: '/ui/web_dashboard/aria-advanced.html', 
            icon: window.LucideIcons?.User || function() { return React.createElement("div", { className: "w-4 h-4" }); }
        }
    ];

    return React.createElement(window.TubelightNavBar, { items: navItems });
};
