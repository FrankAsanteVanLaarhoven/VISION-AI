import http.server
import socketserver
import webbrowser
import os
import time

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run_server():
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving at http://localhost:{PORT}")
            # Open directly to landing-react.html
            webbrowser.open_new_tab(f"http://localhost:{PORT}/landing-react.html")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48: # Address already in use
            print(f"Port {PORT} is already in use. Please close the other application or choose a different port.")
        else:
            raise

if __name__ == "__main__":
    run_server()