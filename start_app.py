import http.server
import socketserver
import webbrowser
import os
import time
import urllib.parse

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        # Handle specific redirects for better UX
        if self.path == '/':
            self.path = '/ui/web_dashboard/index.html'
        elif self.path == '/landing-react.html':
            self.path = '/landing-redirect.html'
        elif self.path == '/dashboard':
            self.path = '/ui/web_dashboard/index.html'
        elif self.path == '/navigation':
            self.path = '/ui/web_dashboard/navigation-controls.html'
        elif self.path == '/aria':
            self.path = '/ui/web_dashboard/aria-advanced.html'
        
        # Handle asset paths for ui/web_dashboard files
        if self.path.startswith('/ui/web_dashboard/') and self.path.endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico')):
            # Check if asset exists in ui/web_dashboard/assets first
            asset_path = self.path.replace('/ui/web_dashboard/', '/ui/web_dashboard/assets/')
            if os.path.exists(os.path.join(DIRECTORY, asset_path.lstrip('/'))):
                self.path = asset_path
            # If not found, try the original path
        
        return super().do_GET()

def run_server():
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving at http://localhost:{PORT}")
            # Open directly to dashboard hub
            webbrowser.open_new_tab(f"http://localhost:{PORT}/")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48: # Address already in use
            print(f"Port {PORT} is already in use. Please close the other application or choose a different port.")
        else:
            raise

if __name__ == "__main__":
    run_server()