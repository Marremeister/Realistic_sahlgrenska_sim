/**
 * simulator.js
 * Hospital transport simulation management
 * Refactored to use HospitalTransport utilities
 */

// Initialize the HospitalTransport library
const transportSystem = HospitalTransport.initialize();

// Keep global reference to SVG for backward compatibility
let svg = null;

// Global state for backward compatibility
let simulationRunning = false;
let transporterNodes = {};

window.onload = function() {
    // Get SVG reference
    svg = d3.select("#hospitalMap");

    // Initialize the simulator view
    transportSystem.graph.load(() => {
        // Render hospital graph
        transportSystem.graph.render('#hospitalMap');

        // Create a clock in the corner of the map
        addMapClock();

        // Load transporters
        transportSystem.transporters.load(() => {
            // Render transporters
            transportSystem.transporters.render('#hospitalMap');
        });

        // Add view toggle button
        addViewToggleButton();
    });

    // Load strategies into dropdown
    loadAvailableStrategies();

    // Load transport table
    transportSystem.requests.updateTransportTable();

    // Hide stop button initially
    document.getElementById("stopSimulationBtn").style.display = "none";

    // Setup event listeners for simulation stats updates - reduced frequency
    setInterval(updateSimulationStats, 5000);

    // Set up socket event handlers from original code that are not yet in transportSystem
    setupSocketHandlers();
};

// Add view toggle button to control panel
function addViewToggleButton() {
    // Find the control panel
    const controlPanel = document.querySelector('.simulator-controls') ||
                        document.querySelector('.control-panel');

    if (!controlPanel) {
        console.warn("Control panel not found for adding view toggle button");
        return;
    }

    // Add the button
    HospitalTransport.ui.addViewToggleButton(controlPanel);

    // Add cluster control elements
    const clusterControlsDiv = document.createElement('div');
    clusterControlsDiv.id = 'clusterControls';
    clusterControlsDiv.style.display = 'none';
    clusterControlsDiv.innerHTML = `
        <div class="control-group">
            <label for="clusterMethodSelect">Cluster Method:</label>
            <select id="clusterMethodSelect" class="form-control">
                <option value="department_type">By Department Type</option>
                <option value="kmeans">K-Means</option>
                <option value="hierarchical">Hierarchical</option>
            </select>
            <button id="applyClusteringBtn" class="btn">Apply</button>
        </div>
        <div class="cluster-stats">
            <span>Clusters: <span id="clusterCount">-</span></span>
        </div>
    `;

    controlPanel.appendChild(clusterControlsDiv);

    // Add event listener to apply clustering button
    document.getElementById('applyClusteringBtn').addEventListener('click', function() {
        const method = document.getElementById('clusterMethodSelect').value;
        HospitalTransport.ui.applyClusterMethod(method);
    });

    // Show/hide cluster controls based on view mode
    document.getElementById('toggleViewModeBtn').addEventListener('click', function() {
        // Wait a bit for the view mode to update
        setTimeout(() => {
            const viewMode = HospitalTransport.state.currentView;
            clusterControlsDiv.style.display = viewMode === 'clustered' ? 'block' : 'none';
        }, 100);
    });
}

// Add a clock to the map (custom functionality not in transport-utils)
function addMapClock() {
    const controlLayer = svg.append("g").attr("class", "control-layer");

    controlLayer.append("rect")
        .attr("x", 20)
        .attr("y", 20)
        .attr("width", 140)
        .attr("height", 30)
        .attr("rx", 5)
        .attr("fill", "rgba(44, 62, 80, 0.8)");

    controlLayer.append("text")
        .attr("id", "map-clock")
        .attr("x", 35)
        .attr("y", 40)
        .attr("fill", "white")
        .attr("font-family", "monospace")
        .attr("font-size", "14px")
        .text("⏱️ 00:00:00");
}

// Setup additional socket handlers beyond what transportSystem provides
function setupSocketHandlers() {
    // Maintain backward compatibility with existing socket handlers
    transportSystem.socket.onEvent('connect', function() {
        console.log('✅ Socket reconnected');
        // Refresh all data on reconnect
        transportSystem.transporters.load();
        transportSystem.requests.updateTransportTable();
    });

    transportSystem.socket.onEvent('disconnect', function() {
        console.log('❌ Socket disconnected');
        transportSystem.log.add("Socket disconnected from server. Waiting for reconnection...", "error");
    });

    // Update map clock when simulator clock ticks
    transportSystem.socket.onEvent("clock_tick", function(data) {
        const formattedTime = formatSimulationClock(data.simTime);

        // Update the in-map clock if it exists
        if (document.getElementById("map-clock")) {
            document.getElementById("map-clock").textContent = formattedTime;
        }
    });
}

// Fetch available assignment strategies for dropdown
function loadAvailableStrategies() {
    transportSystem.ui.populateStrategyDropdown("#strategySelect", (strategies) => {
        console.log("📥 Received strategies:", strategies);
    });
}

// Apply simulator configuration
function applySimulatorConfig() {
    const numTransporters = parseInt(document.getElementById("numTransporters").value);
    const requestInterval = parseInt(document.getElementById("requestInterval").value);
    const strategy = document.getElementById("strategySelect").value;

    transportSystem.simulation.updateConfig({
        num_transporters: numTransporters,
        request_interval: requestInterval,
        strategy: strategy
    }, () => {
        transportSystem.log.add("✅ Configuration updated");

        // Refresh transporter visual display
        transportSystem.transporters.load(() => {
            transportSystem.transporters.render('#hospitalMap');
        });

        // Add a nice visual effect to show config was applied
        const applyBtn = document.getElementById("applyConfigBtn");
        applyBtn.classList.add("btn-flash");
        setTimeout(() => applyBtn.classList.remove("btn-flash"), 500);

        // Update current config display if element exists
        const configDisplay = document.getElementById("currentConfig");
        if (configDisplay) {
            configDisplay.textContent = `${numTransporters} transporters, ${requestInterval}s interval, ${strategy}`;
        }
    });
}

// Start the simulation
function startSimulation() {
    transportSystem.simulation.start(() => {
        simulationRunning = true;
        transportSystem.log.add("▶️ Simulation started");
        document.getElementById("stopSimulationBtn").style.display = "inline-block";
        document.getElementById("startSimulationBtn").style.display = "none";

        // Update simulation status if element exists
        const statusEl = document.getElementById("simulationStatus");
        if (statusEl) {
            statusEl.textContent = "Running";
            statusEl.className = "status-running";
        }
    });
}

// Stop the simulation
function stopSimulation() {
    transportSystem.simulation.stop(() => {
        simulationRunning = false;
        transportSystem.log.add("🛑 Simulation stopped");
        document.getElementById("stopSimulationBtn").style.display = "none";
        document.getElementById("startSimulationBtn").style.display = "inline-block";

        // Update simulation status if element exists
        const statusEl = document.getElementById("simulationStatus");
        if (statusEl) {
            statusEl.textContent = "Stopped";
            statusEl.className = "status-stopped";
        }
    });
}

// Toggle between detailed and clustered view
function toggleViewMode() {
    const currentView = HospitalTransport.state.currentView;
    const newView = currentView === 'detailed' ? 'clustered' : 'detailed';

    // Toggle the view
    HospitalTransport.ui.toggleViewMode(newView);

    // Update cluster controls visibility
    const clusterControls = document.getElementById('clusterControls');
    if (clusterControls) {
        clusterControls.style.display = newView === 'clustered' ? 'block' : 'none';
    }

    // If switching to clustered view and no cluster data, apply default method
    if (newView === 'clustered' &&
        (!HospitalTransport.clusters.data || !Object.keys(HospitalTransport.clusters.data.clusters || {}).length)) {

        HospitalTransport.ui.applyClusterMethod('department_type');
    }
}

// Format simulation time for display
function formatSimulationClock(simTime) {
    return transportSystem.ui.formatTime(simTime);
}

// Update all simulation statistics - but don't interrupt animations
function updateSimulationStats() {
    if (simulationRunning) {
        // Check if any transporters are animating
        const isAnimating = transportSystem.state.animatingTransporters.size > 0;

        if (!isAnimating) {
            // Only refresh transporters if no animations are in progress
            transportSystem.transporters.load(() => {
                transportSystem.transporters.render('#hospitalMap');
            });
        }

        // Always update the table - doesn't affect animations
        transportSystem.requests.updateTransportTable();
    }

    // If we're in clustered view but have no clusters loaded yet, load them
    if (HospitalTransport.state.currentView === 'clustered' &&
        (!HospitalTransport.clusters.data || !Object.keys(HospitalTransport.clusters.data.clusters || {}).length)) {

        HospitalTransport.ui.applyClusterMethod('department_type');
    }
}

// Debug functions from original code
function debugTransporters() {
    console.log("🔍 Current transporters:", Object.keys(transportSystem.state.transporters));
    console.log("🔍 DOM transporter circles:", document.querySelectorAll('.transporter').length);

    // Print status of each transporter
    Object.entries(transportSystem.state.transporters).forEach(([name, transporter]) => {
        console.log(`🔍 Transporter ${name}:`, {
            status: transporter.status,
            location: transporter.current_location,
            animating: transportSystem.state.animatingTransporters.has(name)
        });
    });
}

function refreshAllTransporters() {
    console.log("🔄 Refreshing all transporters...");

    // Clear existing transporters
    svg.select(".transporter-layer").selectAll("*").remove();
    transportSystem.state.animatingTransporters.clear();

    // Reload all transporters
    transportSystem.transporters.load(() => {
        transportSystem.transporters.render('#hospitalMap');
    });

    transportSystem.log.add("Manually refreshed all transporters", "info");
}

// Export functions to global scope for HTML button access
window.applySimulatorConfig = applySimulatorConfig;
window.startSimulation = startSimulation;
window.stopSimulation = stopSimulation;
window.toggleViewMode = toggleViewMode;
window.debugTransporters = debugTransporters;
window.refreshAllTransporters = refreshAllTransporters;
window.loadTransportTable = function() {
    transportSystem.requests.updateTransportTable();
};