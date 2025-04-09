/**
 * main.js
 * Main script for hospital transport playground
 */

// Global variables
let svg = null;

// Initialize when document is ready
window.onload = function() {
    console.log("Window loaded, initializing transport system");

    // Initialize the transport system
    const transportSystem = HospitalTransport.initialize();

    // Get SVG reference
    svg = d3.select("svg");

    // Initialize the view - load graph first, then everything else
    transportSystem.graph.load(() => {
        console.log("Graph loaded successfully");

        // Now render the graph in SVG
        transportSystem.graph.render('svg');

        // Next, populate dropdowns that depend on the graph
        populateDepartmentDropdowns();

        // Load and render transporters on the graph
        transportSystem.transporters.load(() => {
            transportSystem.transporters.render('svg');
        });

        // Load transport requests and update table
        loadTransportRequests();
        loadTransportTable();

        // Set up event handlers
        setupEventHandlers();

        // Set up periodic refresh
        setInterval(updateData, 5000);

        // Add view toggle button to control panel
        addViewToggleButton();
    });

    // Set up toggle table button
    const toggleBtn = document.getElementById("toggleTableBtn");
    if (toggleBtn) {
        toggleBtn.addEventListener("click", function() {
            const container = document.getElementById("transportTableContainer");
            if (container) {
                container.style.display = container.style.display === "none" ? "block" : "none";
            }
        });
    }

    // Hide transport table by default
    const tableContainer = document.getElementById("transportTableContainer");
    if (tableContainer) {
        tableContainer.style.display = "none";
    }
    const viewModeSelect = document.getElementById("viewModeSelect");
if (viewModeSelect) {
  viewModeSelect.addEventListener("change", function() {
    HospitalTransport.ui.toggleViewMode(this.value);
  });
}
};

// Add view toggle button to page
function addViewToggleButton() {
    // Find control panel or create one
    let controlPanel = document.querySelector('.control-panel');

    if (!controlPanel) {
        // Create a control panel if none exists
        controlPanel = document.createElement('div');
        controlPanel.className = 'control-panel';

        // Insert it in a relevant location
        const container = document.querySelector('.visualization-container') ||
                          document.querySelector('.main-container') ||
                          document.body;

        container.appendChild(controlPanel);
    }

    // Add toggle button
    HospitalTransport.ui.addViewToggleButton(controlPanel);

    // Add cluster method selection if it doesn't exist
    if (!document.getElementById('clusterControlsGroup')) {
        const clusterControls = document.createElement('div');
        clusterControls.id = 'clusterControlsGroup';
        clusterControls.style.display = 'none'; // Initially hidden
        clusterControls.innerHTML = `
            <div class="control-group">
                <label for="clusterMethodSelect">Cluster Method:</label>
                <select id="clusterMethodSelect" class="form-control">
                    <option value="department_type">By Department Type</option>
                    <option value="kmeans">K-Means</option>
                    <option value="hierarchical">Hierarchical</option>
                </select>
                <button id="applyClusteringBtn" class="btn">Apply</button>
            </div>
        `;

        controlPanel.appendChild(clusterControls);

        // Add event listener for the apply button
        document.getElementById('applyClusteringBtn').addEventListener('click', function() {
            const method = document.getElementById('clusterMethodSelect').value;
            HospitalTransport.ui.applyClusterMethod(method);
        });

        // Show/hide cluster controls based on view mode
        document.getElementById('toggleViewModeBtn').addEventListener('click', function() {
            // Wait a bit for the view mode to update
            setTimeout(() => {
                const viewMode = HospitalTransport.state.currentView;
                clusterControls.style.display = viewMode === 'clustered' ? 'block' : 'none';
            }, 100);
        });
    }
}

// Set up event handlers for all buttons
function setupEventHandlers() {
    // Add event listeners for all buttons that need them
    addButtonListener("button[onclick='createRequest()']", createRequest);
    addButtonListener("button[onclick='setTransporterStatus()']", setTransporterStatus);
    addButtonListener("button[onclick='addTransporter()']", addTransporter);
    addButtonListener("button[onclick='initiateTransport()']", initiateTransport);
    addButtonListener("button[onclick='returnHome()']", returnHome);
    addButtonListener("button[onclick='simulateOptimally()']", simulateOptimally);
    addButtonListener("button[onclick='simulateRandomly()']", simulateRandomly);
    addButtonListener("button[onclick='stopSimulation()']", stopSimulation);
    addButtonListener("button[onclick='deployOptimization()']", deployOptimization);
    addButtonListener("button[onclick='deployRandomness()']", deployRandomness);

    // Add toggle clustered view button
    addButtonListener("#toggleViewModeBtn", toggleClusteredView);
}

// Toggle between detailed and clustered view
function toggleClusteredView() {
    const currentView = HospitalTransport.state.currentView;
    const newView = currentView === 'detailed' ? 'clustered' : 'detailed';

    HospitalTransport.ui.toggleViewMode(newView);
}

// Helper function to add event listener to a button
function addButtonListener(selector, handler) {
    const btn = document.querySelector(selector);
    if (btn) {
        // Remove inline handler to avoid duplicate calls
        btn.removeAttribute("onclick");
        btn.addEventListener("click", handler);
    }
}

// Populate department dropdowns
function populateDepartmentDropdowns() {
    const originDropdown = document.getElementById("originDropdown");
    const destDropdown = document.getElementById("destinationDropdown");

    if (!originDropdown || !destDropdown) {
        console.log("Department dropdowns not found on page");
        return;
    }

    // Clear existing options
    originDropdown.innerHTML = "";
    destDropdown.innerHTML = "";

    // Get all nodes from the graph
    const graph = HospitalTransport.state.graph;
    if (!graph || !graph.nodes) {
        console.error("Graph data not available");
        return;
    }

    // Add departments to dropdowns
    graph.nodes.forEach(node => {
        if (node.id) {
            // Origin dropdown
            const originOption = document.createElement("option");
            originOption.value = node.id;
            originOption.textContent = node.id;
            originDropdown.appendChild(originOption);

            // Destination dropdown
            const destOption = document.createElement("option");
            destOption.value = node.id;
            destOption.textContent = node.id;
            destDropdown.appendChild(destOption);
        }
    });

    console.log(`Added ${graph.nodes.length} departments to dropdowns`);
}

// Load transport requests and update request dropdown
function loadTransportRequests() {
  const requestSelect = document.getElementById("requestSelect");
  if (!requestSelect) {
    console.warn("Request select dropdown not found");
    return;
  }

  // Clear the dropdown
  requestSelect.innerHTML = '<option value="">Loading requests...</option>';

  // Load requests
  HospitalTransport.requests.load(() => {
    // Clear dropdown
    requestSelect.innerHTML = "";

    const requests = HospitalTransport.state.requests;

    // Check if we have the new format with requests grouped by status
    if (requests.pending && Array.isArray(requests.pending)) {
      console.log("Processing requests in grouped format, found", requests.pending.length, "pending requests");

      // Check if we have any pending requests
      if (requests.pending.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No pending requests";
        requestSelect.appendChild(option);
        return;
      }

      // Add pending requests to dropdown
      requests.pending.forEach((request, index) => {
        const option = document.createElement("option");
        option.value = `pending_${index}`;
        option.textContent = `${request.origin} → ${request.destination} (${request.transport_type || 'stretcher'})`;
        // Store the full request data as an attribute for later use
        option.setAttribute('data-request', JSON.stringify(request));
        requestSelect.appendChild(option);
      });
    } else {
      // Handle legacy format (if needed)
      console.log("No pending requests found or unexpected format");
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No pending requests";
      requestSelect.appendChild(option);
    }
  });
}

// Update transport table with all transport requests
function loadTransportTable() {
    HospitalTransport.requests.updateTransportTable();
}

// Load transporter data
function loadTransporters() {
    const transporterSelect = document.getElementById("transporterSelect");
    if (!transporterSelect) return;

    // Remember current selection
    const currentValue = transporterSelect.value;

    // Clear dropdown
    transporterSelect.innerHTML = "";

    // Load transporters
    HospitalTransport.transporters.load(() => {
        // Add transporters to dropdown
        Object.values(HospitalTransport.state.transporters).forEach(transporter => {
            const option = document.createElement("option");
            option.value = transporter.name;
            option.textContent = `${transporter.name} (${transporter.status})`;
            transporterSelect.appendChild(option);
        });

        // Restore selection if possible
        if (currentValue && [...transporterSelect.options].find(o => o.value === currentValue)) {
            transporterSelect.value = currentValue;
        }
    });
}

// Periodically update data
function updateData() {
    loadTransporters();
    loadTransportRequests();
    loadTransportTable();

    // If we're in clustered view but have no clusters loaded yet, load them
    if (HospitalTransport.state.currentView === 'clustered' &&
        (!HospitalTransport.clusters.data || !Object.keys(HospitalTransport.clusters.data.clusters || {}).length)) {

        HospitalTransport.ui.applyClusterMethod('department_type');
    }
}

// Create a new transport request
function createRequest() {
    const origin = document.getElementById("originDropdown").value;
    const destination = document.getElementById("destinationDropdown").value;
    const transportType = document.getElementById("typeDropdown").value;
    const urgent = document.getElementById("urgentDropdown").value === "true";

    if (!origin || !destination) {
        HospitalTransport.log.add("Please select origin and destination departments", "error");
        return;
    }

    if (origin === destination) {
        HospitalTransport.log.add("Origin and destination cannot be the same", "error");
        return;
    }

    HospitalTransport.requests.createRequest(origin, destination, transportType, urgent, () => {
        HospitalTransport.log.add(`Created request: ${origin} → ${destination}`, "success");
        loadTransportRequests();
        loadTransportTable();
    });
}

// Update transporter status
function setTransporterStatus() {
    const transporter = document.getElementById("transporterSelect").value;
    const status = document.getElementById("statusDropdown").value;

    if (!transporter) {
        HospitalTransport.log.add("Please select a transporter", "error");
        return;
    }

    fetch("/set_transporter_status", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transporter: transporter, status: status })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            HospitalTransport.log.add(`Error: ${data.error}`, "error");
        } else {
            HospitalTransport.log.add(`Changed status of ${transporter} to ${status}`, "success");
            loadTransporters();
            HospitalTransport.transporters.load(() => {
                HospitalTransport.transporters.render('svg');
            });
        }
    })
    .catch(error => {
        HospitalTransport.log.add(`Error: ${error.message}`, "error");
    });
}

// Add a new transporter
function addTransporter() {
    const transporterName = document.getElementById("transporterName").value;

    if (!transporterName) {
        HospitalTransport.log.add("Please enter a transporter name", "error");
        return;
    }

    fetch("/add_transporter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: transporterName })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            HospitalTransport.log.add(`Error: ${data.error}`, "error");
        } else {
            HospitalTransport.log.add(`Added transporter: ${transporterName}`, "success");
            document.getElementById("transporterName").value = "";
            loadTransporters();
        }
    })
    .catch(error => {
        HospitalTransport.log.add(`Error: ${error.message}`, "error");
    });
}

// Initiate transport
function initiateTransport() {
  const transporter = document.getElementById("transporterSelect").value;
  const requestSelect = document.getElementById("requestSelect");
  const selectedOption = requestSelect.options[requestSelect.selectedIndex];
  const requestKey = requestSelect.value;

  if (!transporter) {
    HospitalTransport.log.add("Please select a transporter", "error");
    return;
  }

  if (!requestKey || !selectedOption) {
    HospitalTransport.log.add("Please select a transport request", "error");
    return;
  }

  // Get the request data
  let request;
  if (selectedOption.hasAttribute('data-request')) {
    // New format - get data from the attribute
    try {
      request = JSON.parse(selectedOption.getAttribute('data-request'));
    } catch (error) {
      console.error("Error parsing request data:", error);
      HospitalTransport.log.add("Error retrieving request data", "error");
      return;
    }
  } else {
    // Fallback to old format if needed
    HospitalTransport.log.add("Invalid request format", "error");
    return;
  }

  fetch("/assign_transport", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      transporter: transporter,
      origin: request.origin,
      destination: request.destination,
      transport_type: request.transport_type || "stretcher",
      urgent: request.urgent || false
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      HospitalTransport.log.add(`Error: ${data.error}`, "error");
    } else {
      HospitalTransport.log.add(`Assigned ${transporter} to transport from ${request.origin} to ${request.destination}`, "success");
      loadTransportRequests();
      loadTransportTable();
    }
  })
  .catch(error => {
    HospitalTransport.log.add(`Error: ${error.message}`, "error");
  });
}

// Return transporter to home
function returnHome() {
    const transporter = document.getElementById("transporterSelect").value;

    if (!transporter) {
        HospitalTransport.log.add("Please select a transporter", "error");
        return;
    }

    fetch("/return_home", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transporter: transporter })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            HospitalTransport.log.add(`Error: ${data.error}`, "error");
        } else {
            HospitalTransport.log.add(`${transporter} is returning to home base`, "success");
        }
    })
    .catch(error => {
        HospitalTransport.log.add(`Error: ${error.message}`, "error");
    });
}

// Deploy optimization strategy
function deployOptimization() {
    HospitalTransport.simulation.setStrategy("ILP: Makespan", () => {
        HospitalTransport.simulation.deployStrategy();
    });
}

// Deploy random strategy
function deployRandomness() {
    HospitalTransport.simulation.setStrategy("Random", () => {
        HospitalTransport.simulation.deployStrategy();
    });
}

// Simulate with optimal strategy
function simulateOptimally() {
    HospitalTransport.simulation.setStrategy("ILP: Makespan", () => {
        HospitalTransport.simulation.start();

        // Update UI
        document.getElementById("stopSimulationBtn").style.display = "inline-block";
        document.getElementById("simulateOptimally").style.display = "none";
        document.getElementById("simulateRandomly").style.display = "none";
    });
}

// Simulate with random strategy
function simulateRandomly() {
    HospitalTransport.simulation.setStrategy("Random", () => {
        HospitalTransport.simulation.start();

        // Update UI
        document.getElementById("stopSimulationBtn").style.display = "inline-block";
        document.getElementById("simulateOptimally").style.display = "none";
        document.getElementById("simulateRandomly").style.display = "none";
    });
}

// Stop simulation
function stopSimulation() {
    HospitalTransport.simulation.stop();

    // Update UI
    document.getElementById("stopSimulationBtn").style.display = "none";
    document.getElementById("simulateOptimally").style.display = "inline-block";
    document.getElementById("simulateRandomly").style.display = "inline-block";
}