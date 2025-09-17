// 3D Navigation for QEP-VLA Dashboard with Live Maps

class Navigation3D {
    constructor() {
        this.container = null;
        this.map = null;
        this.vehicles = [];
        this.routes = [];
        this.markers = [];

        // API Tokens (securely configured)
        this.apiTokens = {
            mapbox: 'pk.eyJ1IjoiZmF2MSIsImEiOiJjbTlmNXNkNTQweDg5MmtxdDZ6dHk4MHFrIn0.TVkYeyob4fb0JVFHkcD1zw',
            mappilar: 'MLY|24939769952302534|a72d1d1b67c2c13b435edb54bb3bcf24',
            celsium: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJiNDU2ZTNmMy0zMzczLTQwMTUtYTU0Ni1iZGE4OTVkYmFiNjciLCJpZCI6MzI5MTQ5LCJpYXQiOjE3NTQ0NDA2NDR9.TM0xevdP4PDCSVDtxPMi-lGrCcA5XyVwIT_OChcvp3Y'
        };

        this.currentMapProvider = 'mapbox'; // Default to Mapbox
        this.init();
    }

    init() {
        this.container = document.getElementById('3dNavigation');
        if (!this.container) return;

        this.setup3DScene();
        this.animate();
    }

    setup3DScene() {
        // Create live map navigation interface
        this.container.innerHTML = `
            <div class="3d-navigation-container">
                <div class="navigation-header">
                    <h3>Live Navigation View</h3>
                    <div class="navigation-controls">
                        <select id="mapProvider" class="map-provider-select">
                            <option value="mapbox">Mapbox</option>
                            <option value="celsium">Celsium</option>
                            <option value="mappilar">Mappilar</option>
                        </select>
                        <button class="btn btn-sm btn-primary" onclick="navigation3D.resetView()">Reset View</button>
                        <button class="btn btn-sm btn-secondary" onclick="navigation3D.toggleTraffic()">Traffic</button>
                        <button class="btn btn-sm btn-info" onclick="navigation3D.centerOnFleet()">Center Fleet</button>
                    </div>
                </div>
                <div class="navigation-map-container">
                    <div id="navigationMap" class="navigation-map"></div>
                    <div class="map-overlay">
                        <div class="map-controls">
                            <button class="zoom-in" onclick="navigation3D.zoomIn()">+</button>
                            <button class="zoom-out" onclick="navigation3D.zoomOut()">-</button>
                        </div>
                        <div class="map-legend">
                            <div class="legend-item">
                                <div class="legend-color vehicle"></div>
                                <span>Active Vehicles</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color route"></div>
                                <span>Planned Routes</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color destination"></div>
                                <span>Destinations</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="navigation-info">
                    <div class="info-panel">
                        <h4>Navigation Status</h4>
                        <div class="status-grid">
                            <div class="status-item">
                                <span class="label">Active Routes:</span>
                                <span class="value" id="activeRoutes">3</span>
                            </div>
                            <div class="status-item">
                                <span class="label">Vehicles:</span>
                                <span class="value" id="totalVehicles">25</span>
                            </div>
                            <div class="status-item">
                                <span class="label">Traffic:</span>
                                <span class="value" id="trafficStatus">Low</span>
                            </div>
                        </div>
                    </div>
                    <div class="vehicle-list">
                        <h4>Active Vehicles</h4>
                        <div id="vehicleList" class="vehicle-items">
                            <!-- Vehicle list will be populated here -->
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.loadMapProvider();
        this.setupMapControls();
    }

    loadMapProvider() {
        const provider = this.currentMapProvider;
        console.log(`🗺️ Loading ${provider} map...`);

        switch(provider) {
            case 'mapbox':
                this.loadMapbox();
                break;
            case 'celsium':
                this.loadCelsium();
                break;
            case 'mappilar':
                this.loadMappilar();
                break;
            default:
                this.loadMapbox(); // Default fallback
        }
    }

    loadMapbox() {
        // Load Mapbox GL JS if not already loaded
        if (!window.mapboxgl) {
            const script = document.createElement('script');
            script.src = 'https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js';
            script.onload = () => this.initializeMapbox();
            document.head.appendChild(script);

            // Load CSS
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = 'https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css';
            document.head.appendChild(link);
        } else {
            this.initializeMapbox();
        }
    }

    initializeMapbox() {
        try {
            mapboxgl.accessToken = this.apiTokens.mapbox;

            this.map = new mapboxgl.Map({
                container: 'navigationMap',
                style: 'mapbox://styles/mapbox/streets-v12',
                center: [-74.5, 40], // Default to NYC area
                zoom: 9,
                pitch: 45, // 3D effect
                bearing: -17.6
            });

            this.map.on('load', () => {
                console.log('🗺️ Mapbox map loaded successfully');
                this.addMapLayers();
                this.addVehicleMarkers();
                this.addRouteLayers();
                this.setupMapInteractions();
            });

            this.map.on('error', (e) => {
                console.error('🗺️ Mapbox error:', e);
                this.showMapError('Mapbox failed to load');
            });

        } catch (error) {
            console.error('🗺️ Failed to initialize Mapbox:', error);
            this.showMapError('Mapbox initialization failed');
        }
    }

    loadCelsium() {
        // Celsium Maps integration
        console.log('🗺️ Loading Celsium Maps...');
        // Celsium integration would go here
        this.showMapError('Celsium integration coming soon');
    }

    loadMappilar() {
        // Mappilar integration
        console.log('🗺️ Loading Mappilar...');
        // Mappilar integration would go here
        this.showMapError('Mappilar integration coming soon');
    }

    showMapError(message) {
        const mapContainer = document.getElementById('navigationMap');
        if (mapContainer) {
            mapContainer.innerHTML = `
                <div class="map-error">
                    <h4>Map Loading Error</h4>
                    <p>${message}</p>
                    <button onclick="navigation3D.retryMapLoad()">Retry</button>
                </div>
            `;
        }
    }

    drawNavigationMap(ctx) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw background grid
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 1;
        
        // Vertical lines
        for (let x = 0; x <= width; x += 50) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y <= height; y += 50) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw zones
        this.drawZone(ctx, 100, 100, 150, 100, 'Zone A', '#3498db');
        this.drawZone(ctx, 400, 100, 150, 100, 'Zone B', '#e74c3c');
        this.drawZone(ctx, 250, 300, 150, 100, 'Zone C', '#2ecc71');

        // Draw vehicles
        this.drawVehicles(ctx);

        // Draw routes
        this.drawRoutes(ctx);
    }

    drawZone(ctx, x, y, width, height, label, color) {
        ctx.fillStyle = color + '20';
        ctx.fillRect(x, y, width, height);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
        
        ctx.fillStyle = color;
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(label, x + width/2, y + height/2);
    }

    drawVehicles(ctx) {
        const vehiclePositions = [
            { x: 150, y: 150, id: 'QEP-001', status: 'online' },
            { x: 450, y: 150, id: 'QEP-002', status: 'online' },
            { x: 300, y: 350, id: 'QEP-003', status: 'online' },
            { x: 200, y: 250, id: 'QEP-004', status: 'maintenance' },
            { x: 500, y: 300, id: 'QEP-005', status: 'online' }
        ];

        vehiclePositions.forEach(vehicle => {
            const color = vehicle.status === 'online' ? '#2ecc71' : '#f39c12';
            
            // Vehicle circle
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(vehicle.x, vehicle.y, 8, 0, 2 * Math.PI);
            ctx.fill();
            
            // Vehicle ID
            ctx.fillStyle = '#2c3e50';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(vehicle.id, vehicle.x, vehicle.y + 25);
            
            // Status indicator
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(vehicle.x, vehicle.y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    drawRoutes(ctx) {
        // Draw route lines
        ctx.strokeStyle = '#9b59b6';
        ctx.lineWidth = 3;
        ctx.setLineDash([10, 5]);
        
        // Route 1: Zone A to Zone B
        ctx.beginPath();
        ctx.moveTo(250, 150);
        ctx.lineTo(400, 150);
        ctx.stroke();
        
        // Route 2: Zone B to Zone C
        ctx.beginPath();
        ctx.moveTo(475, 200);
        ctx.lineTo(325, 300);
        ctx.stroke();
        
        // Route 3: Zone A to Zone C
        ctx.beginPath();
        ctx.moveTo(175, 200);
        ctx.lineTo(325, 300);
        ctx.stroke();
        
        ctx.setLineDash([]);
    }

    handleCanvasClick(e) {
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Check if click is on a vehicle
        const clickedVehicle = this.findVehicleAt(x, y);
        if (clickedVehicle) {
            this.showVehicleInfo(clickedVehicle);
        } else {
            // Plan route to clicked location
            this.planRoute(x, y);
        }
    }

    findVehicleAt(x, y) {
        const vehiclePositions = [
            { x: 150, y: 150, id: 'QEP-001', status: 'online' },
            { x: 450, y: 150, id: 'QEP-002', status: 'online' },
            { x: 300, y: 350, id: 'QEP-003', status: 'online' },
            { x: 200, y: 250, id: 'QEP-004', status: 'maintenance' },
            { x: 500, y: 300, id: 'QEP-005', status: 'online' }
        ];

        return vehiclePositions.find(vehicle => {
            const distance = Math.sqrt((x - vehicle.x) ** 2 + (y - vehicle.y) ** 2);
            return distance <= 15;
        });
    }

    showVehicleInfo(vehicle) {
        alert(`Vehicle ${vehicle.id}\nStatus: ${vehicle.status}\nLocation: (${vehicle.x}, ${vehicle.y})`);
    }

    planRoute(x, y) {
        // Simple route planning
        const startZone = this.findNearestZone(150, 150); // Start from Zone A
        const endZone = this.findNearestZone(x, y);
        
        alert(`Route planned from ${startZone} to ${endZone}\nCoordinates: (${x}, ${y})`);
    }

    findNearestZone(x, y) {
        const zones = [
            { name: 'Zone A', x: 175, y: 150 },
            { name: 'Zone B', x: 475, y: 150 },
            { name: 'Zone C', x: 325, y: 350 }
        ];

        let nearest = zones[0];
        let minDistance = Infinity;

        zones.forEach(zone => {
            const distance = Math.sqrt((x - zone.x) ** 2 + (y - zone.y) ** 2);
            if (distance < minDistance) {
                minDistance = distance;
                nearest = zone;
            }
        });

        return nearest.name;
    }

    resetView() {
        this.drawNavigationMap(document.getElementById('navigationCanvas').getContext('2d'));
    }

    toggleWireframe() {
        const canvas = document.getElementById('navigationCanvas');
        const ctx = canvas.getContext('2d');
        
        // Toggle between normal and wireframe view
        if (ctx.strokeStyle === '#2c3e50') {
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 2;
        } else {
            ctx.strokeStyle = '#2c3e50';
            ctx.lineWidth = 1;
        }
        
        this.drawNavigationMap(ctx);
    }

    setupMapControls() {
        // Map provider selector
        const providerSelect = document.getElementById('mapProvider');
        if (providerSelect) {
            providerSelect.addEventListener('change', (e) => {
                this.switchMapProvider(e.target.value);
            });
        }
    }

    switchMapProvider(provider) {
        this.currentMapProvider = provider;
        console.log(`🔄 Switching to ${provider} map provider`);

        // Clear existing map
        if (this.map) {
            this.map.remove();
            this.map = null;
        }

        // Clear markers and routes
        this.clearMarkers();
        this.clearRoutes();

        // Load new provider
        this.loadMapProvider();
    }

    addMapLayers() {
        if (!this.map) return;

        // Add traffic layer if available
        try {
            this.map.addSource('traffic', {
                type: 'vector',
                url: 'mapbox://mapbox.mapbox-traffic-v1'
            });

            this.map.addLayer({
                id: 'traffic-layer',
                type: 'line',
                source: 'traffic',
                'source-layer': 'traffic',
                paint: {
                    'line-color': [
                        'match',
                        ['get', 'congestion'],
                        'low', '#00ff00',
                        'moderate', '#ffff00',
                        'heavy', '#ff0000',
                        'severe', '#8b0000',
                        '#000000'
                    ],
                    'line-width': 2
                }
            }, 'road-label');
        } catch (error) {
            console.warn('Traffic layer not available:', error);
        }
    }

    addVehicleMarkers() {
        if (!this.map) return;

        // Sample vehicle locations (replace with real data)
        const vehicles = [
            { id: 'V001', lng: -74.5, lat: 40.1, status: 'active', speed: 45 },
            { id: 'V002', lng: -74.3, lat: 40.2, status: 'active', speed: 38 },
            { id: 'V003', lng: -74.7, lat: 40.0, status: 'maintenance', speed: 0 },
            { id: 'V004', lng: -74.4, lat: 40.3, status: 'active', speed: 52 },
            { id: 'V005', lng: -74.6, lat: 40.15, status: 'active', speed: 41 }
        ];

        vehicles.forEach(vehicle => {
            this.addVehicleMarker(vehicle);
        });

        this.updateVehicleList(vehicles);
    }

    addVehicleMarker(vehicle) {
        if (!this.map) return;

        const el = document.createElement('div');
        el.className = `vehicle-marker ${vehicle.status}`;
        el.innerHTML = `
            <div class="vehicle-icon">
                <div class="vehicle-pulse"></div>
                <span class="vehicle-id">${vehicle.id.slice(-2)}</span>
            </div>
        `;

        el.addEventListener('click', () => {
            this.showVehicleInfo(vehicle);
        });

        const marker = new mapboxgl.Marker(el)
            .setLngLat([vehicle.lng, vehicle.lat])
            .addTo(this.map);

        this.markers.push(marker);
    }

    addRouteLayers() {
        if (!this.map) return;

        // Sample routes (replace with real route data)
        const routes = [
            {
                id: 'route1',
                coordinates: [
                    [-74.5, 40.1],
                    [-74.4, 40.15],
                    [-74.3, 40.2],
                    [-74.2, 40.25]
                ],
                color: '#ff6b6b'
            },
            {
                id: 'route2',
                coordinates: [
                    [-74.7, 40.0],
                    [-74.65, 40.05],
                    [-74.6, 40.1],
                    [-74.55, 40.15]
                ],
                color: '#4ecdc4'
            }
        ];

        routes.forEach(route => {
            this.addRoute(route);
        });
    }

    addRoute(route) {
        if (!this.map) return;

        this.map.addSource(route.id, {
            type: 'geojson',
            data: {
                type: 'Feature',
                properties: {},
                geometry: {
                    type: 'LineString',
                    coordinates: route.coordinates
                }
            }
        });

        this.map.addLayer({
            id: `${route.id}-layer`,
            type: 'line',
            source: route.id,
            paint: {
                'line-color': route.color,
                'line-width': 4,
                'line-opacity': 0.8
            }
        });

        this.routes.push(route);
    }

    setupMapInteractions() {
        if (!this.map) return;

        // Add click handler for route planning
        this.map.on('click', (e) => {
            this.handleMapClick(e);
        });

        // Add hover effects
        this.map.on('mouseenter', 'traffic-layer', () => {
            this.map.getCanvas().style.cursor = 'pointer';
        });

        this.map.on('mouseleave', 'traffic-layer', () => {
            this.map.getCanvas().style.cursor = '';
        });
    }

    animate() {
        // Update vehicle positions every 3 seconds
        setInterval(() => {
            if (this.map) {
                this.updateVehiclePositions();
                this.updateTrafficStatus();
            }
        }, 3000);
    }

    updateVehiclePositions() {
        // Simulate vehicle movement (replace with real GPS data)
        this.markers.forEach((marker, index) => {
            const lngLat = marker.getLngLat();
            const newLng = lngLat.lng + (Math.random() - 0.5) * 0.001;
            const newLat = lngLat.lat + (Math.random() - 0.5) * 0.001;

            marker.setLngLat([newLng, newLat]);
        });
    }

    updateTrafficStatus() {
        // Simulate traffic updates
        const statuses = ['Low', 'Moderate', 'Heavy', 'Severe'];
        const randomStatus = statuses[Math.floor(Math.random() * statuses.length)];

        const trafficElement = document.getElementById('trafficStatus');
        if (trafficElement) {
            trafficElement.textContent = randomStatus;
            trafficElement.className = `value traffic-${randomStatus.toLowerCase()}`;
        }
    }

    updateVehicleList(vehicles) {
        const vehicleList = document.getElementById('vehicleList');
        if (!vehicleList) return;

        vehicleList.innerHTML = vehicles.map(vehicle => `
            <div class="vehicle-item ${vehicle.status}" onclick="navigation3D.showVehicleInfo(${JSON.stringify(vehicle).replace(/"/g, '&quot;')})">
                <div class="vehicle-avatar">${vehicle.id.slice(-2)}</div>
                <div class="vehicle-details">
                    <div class="vehicle-name">${vehicle.id}</div>
                    <div class="vehicle-status">${vehicle.status} • ${vehicle.speed} km/h</div>
                </div>
            </div>
        `).join('');
    }

    showVehicleInfo(vehicle) {
        // Create popup with vehicle information
        const popup = document.createElement('div');
        popup.className = 'vehicle-popup';
        popup.innerHTML = `
            <div class="popup-header">
                <h4>Vehicle ${vehicle.id}</h4>
                <button onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="popup-content">
                <div class="info-row"><strong>Status:</strong> ${vehicle.status}</div>
                <div class="info-row"><strong>Speed:</strong> ${vehicle.speed} km/h</div>
                <div class="info-row"><strong>Location:</strong> ${vehicle.lng.toFixed(4)}, ${vehicle.lat.toFixed(4)}</div>
                <div class="info-row"><strong>Last Update:</strong> ${new Date().toLocaleTimeString()}</div>
            </div>
        `;

        document.body.appendChild(popup);
    }

    handleMapClick(e) {
        // Handle map clicks for route planning
        console.log('Map clicked at:', e.lngLat);

        // Add destination marker
        const el = document.createElement('div');
        el.className = 'destination-marker';
        el.innerHTML = '📍';

        new mapboxgl.Marker(el)
            .setLngLat(e.lngLat)
            .addTo(this.map);
    }

    clearMarkers() {
        this.markers.forEach(marker => marker.remove());
        this.markers = [];
    }

    clearRoutes() {
        this.routes.forEach(route => {
            if (this.map.getLayer(`${route.id}-layer`)) {
                this.map.removeLayer(`${route.id}-layer`);
            }
            if (this.map.getSource(route.id)) {
                this.map.removeSource(route.id);
            }
        });
        this.routes = [];
    }

    // Control methods
    resetView() {
        if (this.map) {
            this.map.flyTo({
                center: [-74.5, 40],
                zoom: 9,
                pitch: 45,
                bearing: -17.6
            });
        }
    }

    toggleTraffic() {
        if (!this.map) return;

        const trafficLayer = this.map.getLayer('traffic-layer');
        if (trafficLayer) {
            const visibility = this.map.getLayoutProperty('traffic-layer', 'visibility');
            this.map.setLayoutProperty('traffic-layer', 'visibility',
                visibility === 'visible' ? 'none' : 'visible');
        }
    }

    centerOnFleet() {
        if (!this.map || this.markers.length === 0) return;

        // Calculate bounds of all vehicle markers
        const bounds = new mapboxgl.LngLatBounds();
        this.markers.forEach(marker => {
            bounds.extend(marker.getLngLat());
        });

        this.map.fitBounds(bounds, { padding: 50 });
    }

    zoomIn() {
        if (this.map) this.map.zoomIn();
    }

    zoomOut() {
        if (this.map) this.map.zoomOut();
    }

    retryMapLoad() {
        this.loadMapProvider();
    }

}

// Initialize 3D navigation when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.navigation3D = new Navigation3D();
});
