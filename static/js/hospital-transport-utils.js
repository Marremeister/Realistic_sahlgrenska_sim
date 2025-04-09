/**
 * hospital-transport-utils.js
 * Shared utilities for hospital transport visualization system
 */

// Namespace for our utilities
const HospitalTransport = {
  /**
   * Configuration and state
   */
  config: {
    socketUrl: "http://127.0.0.1:5001",
    defaultMapWidth: 1400,
    defaultMapHeight: 1000,
    animationDuration: 1000, // Default animation duration in ms
    logMaxEntries: 100, // Maximum number of log entries to keep
    refreshInterval: 5000, // Interval for auto-refresh in ms
  },

  state: {
    socket: null,
    graph: null,
    transporters: {},
    requests: {},
    simulationRunning: false,
    currentView: "detailed", // "detailed" or "clustered"
    currentCluster: null,
    animatingTransporters: new Set(), // Track which transporters are currently animating
    logBuffer: []
  },

  /**
   * Socket connection and event handling
   */
  socket: {
    initialize: function(url = null) {
      if (HospitalTransport.state.socket) {
        console.warn("Socket already initialized");
        return HospitalTransport.state.socket;
      }

      const socketUrl = url || HospitalTransport.config.socketUrl;
      HospitalTransport.state.socket = io(socketUrl);

      // Setup core event listeners
      HospitalTransport._setupSocketListeners();

      console.log("Socket initialized to:", socketUrl);
      return HospitalTransport.state.socket;
    },

    emit: function(event, data) {
      if (!HospitalTransport.state.socket) {
        console.error("Socket not initialized. Call HospitalTransport.socket.initialize() first");
        return;
      }

      HospitalTransport.state.socket.emit(event, data);
    },

    onEvent: function(event, callback) {
      if (!HospitalTransport.state.socket) {
        console.error("Socket not initialized. Call HospitalTransport.socket.initialize() first");
        return;
      }

      HospitalTransport.state.socket.on(event, callback);
    }
  },

  /**
   * Graph visualization utilities
   */
  graph: {
    load: function(callback) {
      fetch(`${HospitalTransport.config.socketUrl}/get_hospital_graph`)
        .then(response => response.json())
        .then(graph => {
          HospitalTransport.state.graph = graph;
          if (callback) callback(graph);
        })
        .catch(error => {
          console.error("Error loading hospital graph:", error);
          HospitalTransport.log.add("Error loading hospital graph", "error");
        });
    },

    render: function(svgSelector, options = {}) {
      const svg = d3.select(svgSelector);

      if (!HospitalTransport.state.graph) {
        console.error("Graph not loaded. Call HospitalTransport.graph.load() first");
        return;
      }

      // Clear existing elements if specified
      if (options.clear !== false) {
        svg.selectAll("*").remove();
      }

      const graph = HospitalTransport.state.graph;
      const nodesMap = new Map(graph.nodes.map(n => [n.id, { id: n.id, x: n.x, y: n.y }]));

      const edges = graph.edges.map(e => ({
        source: nodesMap.get(e.source),
        target: nodesMap.get(e.target),
        weight: e.distance
      }));

      // Default options
      const defaults = {
        showEdgeWeights: true,
        nodeRadius: 20,
        nodeColor: "#90CAF9",
        edgeColor: "#bbb",
        labelOffset: { x: 10, y: -10 },
        groups: null // Optional grouping info for nodes
      };

      // Merge options with defaults
      const settings = {...defaults, ...options};

      // Create layer groups
      const edgeLayer = svg.append("g").attr("class", "edge-layer");
      const nodeLayer = svg.append("g").attr("class", "node-layer");
      const labelLayer = svg.append("g").attr("class", "label-layer");
      const transporterLayer = svg.append("g").attr("class", "transporter-layer");

      // Draw edges (connections)
      edgeLayer.selectAll("line")
        .data(edges)
        .enter()
        .append("line")
        .attr("class", "link")
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y)
        .style("stroke", settings.edgeColor)
        .style("stroke-width", "2px");

      // Draw edge weights if enabled
      if (settings.showEdgeWeights) {
        edgeLayer.selectAll("text.edge-label")
          .data(edges)
          .enter()
          .append("text")
          .attr("class", "edge-label")
          .attr("x", d => (d.source.x + d.target.x) / 2)
          .attr("y", d => (d.source.y + d.target.y) / 2 - 5)
          .attr("text-anchor", "middle")
          .attr("fill", "#777")
          .attr("font-size", "12px")
          .attr("font-weight", "bold")
          .text(d => d.weight);
      }

      // Draw nodes (departments)
      const nodes = nodeLayer.selectAll("circle")
        .data([...nodesMap.values()])
        .enter()
        .append("circle")
        .attr("class", "node")
        .attr("r", settings.nodeRadius)
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("data-node", d => d.id)
        .style("fill", d => {
          // If groups are specified, color by group
          if (settings.groups && settings.groups[d.id]) {
            return settings.groups[d.id].color || settings.nodeColor;
          }
          return settings.nodeColor;
        })
        .style("stroke", "#2c3e50")
        .style("stroke-width", "2px");

      // Add node interactions if any
      if (options.onNodeClick) {
        nodes.style("cursor", "pointer")
          .on("click", (event, d) => options.onNodeClick(d));
      }

      // Draw node labels
      labelLayer.selectAll("text")
        .data([...nodesMap.values()])
        .enter()
        .append("text")
        .attr("class", "node-label")
        .attr("x", d => d.x + settings.labelOffset.x)
        .attr("y", d => d.y + settings.labelOffset.y)
        .text(d => {
          // Show short or full node name based on settings
          if (settings.shortLabels) {
            return d.id.substring(0, 3);
          }
          return d.id;
        })
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .style("fill", "#333");

      // Return created layers for future reference
      return {
        edgeLayer,
        nodeLayer,
        labelLayer,
        transporterLayer,
        nodesMap
      };
    },

    getNodePosition: function(nodeId) {
      if (!HospitalTransport.state.graph) {
        console.error("Graph not loaded. Call HospitalTransport.graph.load() first");
        return null;
      }

      const node = HospitalTransport.state.graph.nodes.find(n => n.id === nodeId);
      if (!node) {
        console.warn(`Node ${nodeId} not found in graph`);
        return null;
      }

      return { x: node.x, y: node.y };
    }
  },

  /**
   * Transporter visualization and management
   */
  transporters: {
    load: function(callback) {
      fetch(`${HospitalTransport.config.socketUrl}/get_transporters`)
        .then(response => response.json())
        .then(transporters => {
          // Update state with transporters
          transporters.forEach(t => {
            HospitalTransport.state.transporters[t.name] = t;
          });

          if (callback) callback(transporters);
        })
        .catch(error => {
          console.error("Error loading transporters:", error);
          HospitalTransport.log.add("Error loading transporters", "error");
        });
    },

    render: function(svgSelector, options = {}) {
      const svg = d3.select(svgSelector);
      let transporterLayer = svg.select(".transporter-layer");

      // If transporter layer doesn't exist, create it
      if (transporterLayer.empty()) {
        transporterLayer = svg.append("g").attr("class", "transporter-layer");
      }

      // Clear existing transporters if requested
      if (options.clear) {
        transporterLayer.selectAll("*").remove();
      }

      // Get all transporters
      const transporters = Object.values(HospitalTransport.state.transporters);
      console.log(`Rendering ${transporters.length} transporters`);

      // Render each transporter
      transporters.forEach(transporter => {
        // Skip if transporter is animating
        if (HospitalTransport.state.animatingTransporters.has(transporter.name)) {
          return;
        }

        // Skip if filtering by departments and not in the filter list
        if (options.filterDepartments &&
            !options.filterDepartments.includes(transporter.current_location)) {
          // Remove any existing transporter elements that are no longer visible
          transporterLayer.selectAll(`[data-transporter="${transporter.name}"]`).remove();
          transporterLayer.selectAll(`[data-transporter-label="${transporter.name}"]`).remove();
          return;
        }

        // Remove existing transporter elements
        transporterLayer.selectAll(`[data-transporter="${transporter.name}"]`).remove();
        transporterLayer.selectAll(`[data-transporter-label="${transporter.name}"]`).remove();

        // Determine position based on options
        let xPos, yPos;

        if (options.positionByCluster && HospitalTransport.clusters.data) {
          // Position by cluster center
          const clusterID = HospitalTransport.clusters.getClusterForDepartment(transporter.current_location);
          if (clusterID && HospitalTransport.clusters.data.clusters[clusterID]) {
            const center = HospitalTransport.clusters.data.clusters[clusterID].center;
            xPos = center[0];
            yPos = center[1];

            // Add some random jitter to avoid transporters stacking exactly on top of each other
            xPos += (Math.random() - 0.5) * 30;
            yPos += (Math.random() - 0.5) * 30;
          } else {
            // Fall back to department position if cluster not found
            const nodePos = HospitalTransport.graph.getNodePosition(transporter.current_location);
            if (!nodePos) return; // Skip if position not found
            xPos = nodePos.x;
            yPos = nodePos.y;
          }
        } else {
          // Position by department
          const nodePos = HospitalTransport.graph.getNodePosition(transporter.current_location);
          if (!nodePos) {
            console.warn(`Position not found for ${transporter.current_location}`);
            return; // Skip if position not found
          }
          xPos = nodePos.x;
          yPos = nodePos.y;
        }

        // Determine color based on status
        const color = transporter.status === "active" ? "#FF5252" : "#A9A9A9";
        const radius = options.radius || 15;

        // Create transporter circle
        const transporterCircle = transporterLayer.append("circle")
          .attr("class", "transporter")
          .attr("r", radius)
          .attr("cx", xPos)
          .attr("cy", yPos)
          .attr("fill", color)
          .attr("stroke", "white")
          .attr("stroke-width", 2)
          .attr("data-transporter", transporter.name)
          .attr("data-animating", "false")
          .attr("data-current-location", transporter.current_location);

        // Add label if not disabled
        if (options.showLabels !== false) {
          transporterLayer.append("text")
            .attr("class", "transporter-label")
            .attr("x", xPos)
            .attr("y", yPos - 20)
            .attr("text-anchor", "middle")
            .attr("font-size", "14px")
            .attr("font-weight", "bold")
            .attr("fill", "#333")
            .attr("stroke", "white")
            .attr("stroke-width", "0.5")
            .attr("data-transporter-label", transporter.name)
            .text(transporter.name.replace(/^Sim_Transporter_/, "T"));
        }
      });

      return transporterLayer;
    },

    animatePath: function(transporterName, path, durations, svgSelector = null) {
      // If no SVG selector provided, try to find SVG by default
      const svg = svgSelector ? d3.select(svgSelector) : d3.select("svg");

      if (svg.empty()) {
        console.error("SVG element not found for transporter animation");
        return;
      }

      // Validate path
      if (!path || path.length < 2) {
        console.warn(`Invalid path for ${transporterName}:`, path);
        return;
      }

      // Mark transporter as animating
      HospitalTransport.state.animatingTransporters.add(transporterName);

      // Get transporter elements
      const transporterCircle = svg.select(`[data-transporter="${transporterName}"]`);
      const transporterLabel = svg.select(`[data-transporter-label="${transporterName}"]`);

      if (transporterCircle.empty()) {
        console.error(`Transporter ${transporterName} not found in SVG`);
        HospitalTransport.state.animatingTransporters.delete(transporterName);
        return;
      }

      // Mark as animating
      transporterCircle.attr("data-animating", "true");
      transporterCircle.attr("data-current-location", path[0]);

      // Animation function for each step
      let step = 1;
      function moveStep() {
        if (step >= path.length) {
          // Animation complete
          transporterCircle.attr("data-animating", "false");
          transporterCircle.attr("data-current-location", path[path.length - 1]);
          HospitalTransport.state.animatingTransporters.delete(transporterName);
          return;
        }

        // Get position of next node
        let nextX, nextY;

        // Handle different view modes
        if (HospitalTransport.state.currentView === "clustered" && !HospitalTransport.state.currentCluster) {
          // In cluster overview - get cluster center
          const clusterID = path[step]; // In cluster view, path contains cluster IDs
          if (HospitalTransport.clusters.data &&
              HospitalTransport.clusters.data.clusters &&
              HospitalTransport.clusters.data.clusters[clusterID]) {
            const center = HospitalTransport.clusters.data.clusters[clusterID].center;
            nextX = center[0];
            nextY = center[1];
          } else {
            console.error(`Cluster ${path[step]} not found for transporter animation`);
            transporterCircle.attr("data-animating", "false");
            HospitalTransport.state.animatingTransporters.delete(transporterName);
            return;
          }
        } else {
          // In detailed view or inside a cluster - get department position
          const nextNodePosition = HospitalTransport.graph.getNodePosition(path[step]);
          if (!nextNodePosition) {
            console.error(`Node ${path[step]} not found for transporter animation`);
            transporterCircle.attr("data-animating", "false");
            HospitalTransport.state.animatingTransporters.delete(transporterName);
            return;
          }
          nextX = nextNodePosition.x;
          nextY = nextNodePosition.y;
        }

        // Duration for this step (default to 1000ms if not provided)
        const duration = (durations && durations[step - 1]) || 1000;

        // Animate circle
        transporterCircle
          .transition()
          .duration(duration)
          .attr("cx", nextX)
          .attr("cy", nextY);

        // Animate label if it exists
        if (!transporterLabel.empty()) {
          transporterLabel
            .transition()
            .duration(duration)
            .attr("x", nextX)
            .attr("y", nextY - 20);
        }

        // Move to next step after animation completes
        transporterCircle
          .transition()
          .duration(duration)
          .on("end", () => {
            transporterCircle.attr("data-current-location", path[step]);
            step++;
            moveStep();
          });
      }

      // Start animation
      moveStep();
    }
  },

  /**
   * Transport request handling
   */
  requests: {
    load: function(callback) {
      fetch(`${HospitalTransport.config.socketUrl}/get_transport_requests`)
        .then(response => {
          if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
          }
          return response.json();
        })
        .then(requests => {
          // Store requests in state
          HospitalTransport.state.requests = requests;

          console.log("Loaded transport requests:", requests);

          // Call callback with the requests data
          if (callback) callback(requests);
        })
        .catch(error => {
          console.error("Error loading transport requests:", error);
          HospitalTransport.log.add(`Error loading transport requests: ${error.message}`, "error");

          // Initialize with empty object on error
          HospitalTransport.state.requests = {};

          // Still call callback with empty data
          if (callback) callback({});
        });
    },

    loadAll: function(callback) {
      fetch(`${HospitalTransport.config.socketUrl}/get_all_transports`)
        .then(response => response.json())
        .then(transports => {
          if (callback) callback(transports);
        })
        .catch(error => {
          console.error("Error loading all transports:", error);
          HospitalTransport.log.add("Error loading all transports", "error");

          // Call callback with empty array on error
          if (callback) callback([]);
        });
    },

    updateTransportTable: function(tableId = "transportTable") {
      const table = document.getElementById(tableId);
      if (!table) {
        console.warn(`Transport table with ID ${tableId} not found`);
        return;
      }

      HospitalTransport.requests.loadAll(transports => {
        const tbody = table.querySelector("tbody");
        if (!tbody) return;

        tbody.innerHTML = "";

        if (!transports || transports.length === 0) {
          // Add a "no transports" row
          const row = document.createElement("tr");
          row.innerHTML = '<td colspan="6" style="text-align: center;">No transport requests found</td>';
          tbody.appendChild(row);
          return;
        }

        transports.forEach(req => {
          const row = document.createElement("tr");
          row.innerHTML = `
            <td>${req.origin || ''}</td>
            <td>${req.destination || ''}</td>
            <td>${req.transport_type || ''}</td>
            <td>${req.urgent ? "Yes" : "No"}</td>
            <td>${req.assigned_transporter || ''}</td>
            <td class="status-${req.status || 'unknown'}">${req.status || 'Unknown'}</td>
          `;
          tbody.appendChild(row);
        });

        // Update counters if they exist
        HospitalTransport.requests.updateCounters(transports);
      });
    },

    updateCounters: function(transports) {
      // Update pending counter
      const pendingCounter = document.getElementById("pendingRequests");
      if (pendingCounter) {
        const pendingCount = transports.filter(r => r.status === "pending").length;
        pendingCounter.textContent = pendingCount;
      }

      // Update completed counter
      const completedCounter = document.getElementById("completedRequests");
      if (completedCounter) {
        const completedCount = transports.filter(r => r.status === "completed").length;
        completedCounter.textContent = completedCount;
      }

      // Update active transporters counter
      const activeCounter = document.getElementById("activeTransporters");
      if (activeCounter) {
        HospitalTransport.transporters.load(transporters => {
          const activeCount = transporters.filter(t => t.status === "active").length;
          activeCounter.textContent = activeCount;
        });
      }
    },

    createRequest: function(origin, destination, transportType = "stretcher", urgent = false, callback) {
      fetch(`${HospitalTransport.config.socketUrl}/frontend_transport_request`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin: origin,
          destination: destination,
          transport_type: transportType,
          urgent: urgent
        })
      })
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          HospitalTransport.log.add(`Error creating request: ${data.error}`, "error");
        } else {
          HospitalTransport.log.add(`Created transport request: ${origin} → ${destination}`, "success");
        }

        if (callback) callback(data);
      })
      .catch(error => {
        console.error("Error creating request:", error);
        HospitalTransport.log.add("Error creating transport request", "error");

        if (callback) callback({error: error.message});
      });
    }
  },

  /**
   * Simulation control
   */
  simulation: {
    start: function(callback) {
      fetch(`${HospitalTransport.config.socketUrl}/toggle_simulation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ running: true })
      })
      .then(response => response.json())
      .then(data => {
        HospitalTransport.state.simulationRunning = true;
        HospitalTransport.log.add("▶️ Simulation started", "success");

        // Update UI elements if they exist
        const startBtn = document.getElementById("startSimulationBtn");
        const stopBtn = document.getElementById("stopSimulationBtn");

        if (startBtn) startBtn.style.display = "none";
        if (stopBtn) stopBtn.style.display = "inline-block";

        // Update simulation status
        const statusEl = document.getElementById("simulationStatus");
        if (statusEl) {
          statusEl.textContent = "Running";
          statusEl.className = "status-running";
        }

        if (callback) callback(data);
      })
      .catch(error => {
        console.error("Error starting simulation:", error);
        HospitalTransport.log.add("Error starting simulation", "error");

        if (callback) callback({error: error.message});
      });
    },

    stop: function(callback) {
      fetch(`${HospitalTransport.config.socketUrl}/toggle_simulation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ running: false })
      })
      .then(response => response.json())
      .then(data => {
        HospitalTransport.state.simulationRunning = false;
        HospitalTransport.log.add("🛑 Simulation stopped", "success");

        // Update UI elements if they exist
        const startBtn = document.getElementById("startSimulationBtn");
        const stopBtn = document.getElementById("stopSimulationBtn");

        if (startBtn) startBtn.style.display = "inline-block";
        if (stopBtn) stopBtn.style.display = "none";

        // Update simulation status
        const statusEl = document.getElementById("simulationStatus");
        if (statusEl) {
          statusEl.textContent = "Stopped";
          statusEl.className = "status-stopped";
        }

        if (callback) callback(data);
      })
      .catch(error => {
        console.error("Error stopping simulation:", error);
        HospitalTransport.log.add("Error stopping simulation", "error");

        if (callback) callback({error: error.message});
      });
    },

    setStrategy: function(strategyName, callback) {
      fetch(`${HospitalTransport.config.socketUrl}/set_strategy_by_name`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: strategyName })
      })
      .then(response => response.json())
      .then(data => {
        HospitalTransport.log.add(`Strategy set to ${strategyName}`, "success");
        if (callback) callback(data);
      })
      .catch(error => {
        console.error("Error setting strategy:", error);
        HospitalTransport.log.add("Error setting strategy", "error");

        if (callback) callback({error: error.message});
      });
    },

    deployStrategy: function(callback) {
      fetch(`${HospitalTransport.config.socketUrl}/deploy_strategy_assignment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      })
      .then(response => response.json())
      .then(data => {
        HospitalTransport.log.add("Strategy deployment initiated", "success");
        if (callback) callback(data);
      })
      .catch(error => {
        console.error("Error deploying strategy:", error);
        HospitalTransport.log.add("Error deploying strategy", "error");

        if (callback) callback({error: error.message});
      });
    },

    updateConfig: function(config, callback) {
      fetch(`${HospitalTransport.config.socketUrl}/update_simulator_config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config)
      })
      .then(response => response.json())
      .then(data => {
        HospitalTransport.log.add("Simulator configuration updated", "success");
        if (callback) callback(data);
      })
      .catch(error => {
        console.error("Error updating simulator config:", error);
        HospitalTransport.log.add("Error updating simulator configuration", "error");

        if (callback) callback({error: error.message});
      });
    }
  },

  /**
   * UI utilities
   */
  ui: {
    populateDepartmentDropdowns: function(selectorOrigin, selectorDestination) {
      const originDropdown = document.querySelector(selectorOrigin);
      const destDropdown = document.querySelector(selectorDestination);

      if (!originDropdown || !destDropdown) {
        console.warn("Dropdown elements not found");
        return;
      }

      // If graph is not loaded, load it first
      if (!HospitalTransport.state.graph) {
        console.warn("Graph not loaded, cannot populate dropdowns");
        return;
      }

      // Clear existing options
      originDropdown.innerHTML = "";
      destDropdown.innerHTML = "";

      // Get departments from graph
      const departments = HospitalTransport.state.graph.nodes.map(node => node.id);

      if (departments.length === 0) {
        console.warn("No departments found in graph");
        // Add a placeholder option
        const placeholderOption = document.createElement("option");
        placeholderOption.value = "";
        placeholderOption.textContent = "No departments available";
        originDropdown.appendChild(placeholderOption.cloneNode(true));
        destDropdown.appendChild(placeholderOption);
        return;
      }

      // Add departments to dropdowns
      departments.forEach(dept => {
        const originOption = document.createElement("option");
        originOption.value = dept;
        originOption.textContent = dept;
        originDropdown.appendChild(originOption);

        const destOption = document.createElement("option");
        destOption.value = dept;
        destOption.textContent = dept;
        destDropdown.appendChild(destOption);
      });

      console.log(`Added ${departments.length} departments to dropdowns`);
    },

    populateStrategyDropdown: function(selector, callback) {
      fetch(`${HospitalTransport.config.socketUrl}/get_available_strategies`)
        .then(response => response.json())
        .then(strategies => {
          const dropdown = document.querySelector(selector);
          if (!dropdown) {
            console.warn("Strategy dropdown element not found");
            return;
          }

          // Clear existing options
          dropdown.innerHTML = "";

          // Create and add options
          strategies.forEach(strategy => {
            const option = document.createElement("option");
            option.value = strategy;
            option.textContent = strategy;
            dropdown.appendChild(option);
          });

          if (callback) callback(strategies);
        })
        .catch(error => {
          console.error("Error loading strategies:", error);
          HospitalTransport.log.add("Error loading available strategies", "error");
        });
    },

    formatTime: function(seconds) {
      if (!seconds) return '00:00:00';

      const hrs = Math.floor(seconds / 3600);
      const mins = Math.floor((seconds % 3600) / 60);
      const secs = Math.floor(seconds % 60);

      return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    },

    updateClock: function(simTime, selector = "#sim-clock") {
      const clockElement = document.querySelector(selector);
      if (!clockElement) return;

      const formattedTime = HospitalTransport.ui.formatTime(simTime);
      clockElement.textContent = `⏱️ ${formattedTime}`;
    },

    /**
     * Toggle between detailed and clustered view modes
     */
    toggleViewMode: function(mode) {
      // Update view mode in state
      const previousMode = HospitalTransport.state.currentView;
      HospitalTransport.state.currentView = mode;

      // Log the view change
      HospitalTransport.log.add(`Switching to ${mode} view`, "info");

      if (mode === 'clustered') {
        // Load clusters and display
        HospitalTransport.clusters.load((clusterData) => {
          if (!clusterData) {
            HospitalTransport.log.add("No cluster data available. Generating clusters...", "info");
            // Apply default clustering
            this.applyClusterMethod("department_type");
          } else {
            // Render cluster view
            HospitalTransport.clusters.renderClusterView('svg', {
              clear: true,
              onClusterClick: (clusterId, clusterData) => {
                HospitalTransport.clusters.enterCluster(clusterId, clusterData);
              }
            });

            // Render transporters positioned by cluster
            HospitalTransport.transporters.load(() => {
              HospitalTransport.transporters.render('svg', {
                positionByCluster: true
              });
            });
          }
        });

        // Update any UI controls if they exist
        const viewModeSelect = document.getElementById('viewModeSelect');
        if (viewModeSelect) viewModeSelect.value = 'clustered';

        const toggleBtn = document.getElementById('toggleViewModeBtn');
        if (toggleBtn) toggleBtn.textContent = 'Switch to Detailed View';
      }
      else if (mode === 'detailed') {
        // Clear any current cluster selection
        HospitalTransport.state.currentCluster = null;

        // Ensure graph is loaded
        if (!HospitalTransport.state.graph) {
          HospitalTransport.log.add("Loading graph for detailed view...", "info");
          HospitalTransport.graph.load(() => this._renderDetailedView());
        } else {
          this._renderDetailedView();
        }

        // Update any UI controls if they exist
        const viewModeSelect = document.getElementById('viewModeSelect');
        if (viewModeSelect) viewModeSelect.value = 'detailed';

        const toggleBtn = document.getElementById('toggleViewModeBtn');
        if (toggleBtn) toggleBtn.textContent = 'Switch to Clustered View';
      }

      // Return previous mode for tracking changes
      return previousMode;
    },

    /**
     * Helper function for rendering detailed view
     */
    _renderDetailedView: function() {
      // Clear SVG and render graph
      const svg = d3.select('svg');
      svg.selectAll("*").remove();

      // Render graph with all nodes and edges
      HospitalTransport.graph.render('svg');

      // Render transporters at their current locations
      HospitalTransport.transporters.load(() => {
        HospitalTransport.transporters.render('svg');
      });
    },

    /**
     * Apply a specific clustering method
     */
    applyClusterMethod: function(method) {
      // Show loading message
      HospitalTransport.log.add(`Applying ${method} clustering...`, "info");

      // Call backend to apply clustering
      fetch(`${HospitalTransport.config.socketUrl}/apply_clustering`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method: method })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (data.error) {
          HospitalTransport.log.add(`Error: ${data.error}`, "error");
          return;
        }

        HospitalTransport.log.add(`Cluster method applied: ${method}`, "success");

        // Update UI elements if they exist
        const currentMethodEl = document.getElementById('currentClusterMethod');
        if (currentMethodEl) currentMethodEl.textContent = method;

        // Load clusters and render
        HospitalTransport.clusters.load((clusterData) => {
          // Update cluster count if element exists
          const clusterCountEl = document.getElementById('clusterCount');
          if (clusterCountEl && clusterData && clusterData.clusters) {
            clusterCountEl.textContent = Object.keys(clusterData.clusters).length;
          }

          // Render cluster view
          HospitalTransport.clusters.renderClusterView('svg', {
            clear: true,
            onClusterClick: (clusterId, clusterData) => {
              HospitalTransport.clusters.enterCluster(clusterId, clusterData);
            }
          });

          // Render transporters
          HospitalTransport.transporters.load(() => {
            HospitalTransport.transporters.render('svg', {
              positionByCluster: true
            });
          });
        });
      })
      .catch(error => {
        console.error("Error applying clustering method:", error);
        HospitalTransport.log.add(`Clustering failed: ${error.message}`, "error");

        // Fall back to detailed view on error
        HospitalTransport.ui.toggleViewMode('detailed');
      });
    },

    /**
     * Add toggle button for switching between detailed and clustered views
     */
    addViewToggleButton: function(container) {
      if (!container) {
        console.warn("Container not found for view toggle button");
        return;
      }

      // Create button if it doesn't exist
      if (!document.getElementById('toggleViewModeBtn')) {
        const button = document.createElement('button');
        button.id = 'toggleViewModeBtn';
        button.className = 'btn';
        button.textContent = 'Switch to Clustered View';
        button.addEventListener('click', () => {
          const newMode = HospitalTransport.state.currentView === 'detailed' ? 'clustered' : 'detailed';
          HospitalTransport.ui.toggleViewMode(newMode);
        });

        container.appendChild(button);
      }
    }
  },

  /**
   * Logging utilities
   */
  log: {
    buffer: [],

    add: function(message, type = "info") {
      // Create log entry
      const entry = {
        message,
        type,
        timestamp: new Date().toLocaleTimeString()
      };

      // Add to buffer
      HospitalTransport.state.logBuffer.push(entry);

      // Trim buffer if needed
      if (HospitalTransport.state.logBuffer.length > HospitalTransport.config.logMaxEntries) {
        HospitalTransport.state.logBuffer.shift();
      }

      // Update log display if exists
      HospitalTransport.log.updateDisplay();

      return entry;
    },

    updateDisplay: function(selector = "#logList") {
      const logElement = document.querySelector(selector);
      if (!logElement) return;

      // Add new entries
      HospitalTransport.state.logBuffer.forEach(entry => {
        // Check if entry already exists
        const existingEntries = Array.from(logElement.children);
        const exists = existingEntries.some(el =>
          el.textContent.includes(entry.message) &&
          el.textContent.includes(entry.timestamp)
        );

        if (!exists) {
          const logItem = document.createElement("li");
          logItem.textContent = `[${entry.timestamp}] ${entry.message}`;

          // Apply styles based on type
          if (entry.type === "error") {
            logItem.style.color = "red";
          } else if (entry.type === "success") {
            logItem.style.color = "green";
          }

          logElement.appendChild(logItem);

          // Scroll to bottom
          logElement.scrollTop = logElement.scrollHeight;
        }
      });

      // Trim log display if too long
      while (logElement.children.length > HospitalTransport.config.logMaxEntries) {
        logElement.removeChild(logElement.firstChild);
      }
    },

    clear: function(selector = "#logList") {
      const logElement = document.querySelector(selector);
      if (logElement) {
        logElement.innerHTML = "";
      }

      HospitalTransport.state.logBuffer = [];
    }
  },

  /**
   * Clustering visualization support
   */
  clusters: {
    data: null,

    load: function(callback) {
      fetch(`${HospitalTransport.config.socketUrl}/get_hospital_clusters`)
        .then(response => {
          if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
          }
          return response.json();
        })
        .then(clusters => {
          HospitalTransport.clusters.data = clusters;
          if (callback) callback(clusters);
        })
        .catch(error => {
          console.error("Error loading hospital clusters:", error);
          HospitalTransport.log.add("Error loading hospital clusters", "error");

          // Call callback with null on error
          if (callback) callback(null);
        });
    },

    getClusterForDepartment: function(departmentId) {
      if (!HospitalTransport.clusters.data || !HospitalTransport.clusters.data.department_to_cluster) {
        return null;
      }

      return HospitalTransport.clusters.data.department_to_cluster[departmentId];
    },

    renderClusterView: function(svgSelector, options = {}) {
      if (!HospitalTransport.clusters.data) {
        console.error("Cluster data not loaded. Call HospitalTransport.clusters.load() first");
        return null;
      }

      const svg = d3.select(svgSelector);

      // Clear existing content if specified
      if (options.clear !== false) {
        svg.selectAll("*").remove();
      }

      const clusters = HospitalTransport.clusters.data.clusters;
      const clusterEntries = Object.entries(clusters);

      // Create layer groups
      const linkLayer = svg.append("g").attr("class", "cluster-link-layer");
      const clusterLayer = svg.append("g").attr("class", "cluster-node-layer");
      const labelLayer = svg.append("g").attr("class", "cluster-label-layer");
      const transporterLayer = svg.append("g").attr("class", "transporter-layer");

      // Draw inter-cluster connections if available
      if (HospitalTransport.clusters.data.connections) {
        linkLayer.selectAll("line")
          .data(HospitalTransport.clusters.data.connections)
          .enter()
          .append("line")
          .attr("class", "cluster-link")
          .attr("x1", d => clusters[d.source].center[0])
          .attr("y1", d => clusters[d.source].center[1])
          .attr("x2", d => clusters[d.target].center[0])
          .attr("y2", d => clusters[d.target].center[1])
          .style("stroke", "#bbb")
          .style("stroke-width", d => Math.max(1, Math.min(8, d.strength / 5)))
          .style("stroke-opacity", 0.6);
      }

      // Draw cluster nodes
      const clusterNodes = clusterLayer.selectAll(".cluster-node")
        .data(clusterEntries)
        .enter()
        .append("circle")
        .attr("class", "cluster-node node")
        .attr("cx", d => d[1].center[0])
        .attr("cy", d => d[1].center[1])
        .attr("r", d => Math.sqrt(d[1].size) * 5)  // Size based on number of departments
        .attr("fill", d => this._getClusterColor(d[1].dominant_type))
        .attr("stroke", "#2c3e50")
        .attr("stroke-width", "2px")
        .attr("data-cluster-id", d => d[0]);

      // Add click handler if provided
      if (options.onClusterClick) {
        clusterNodes
          .style("cursor", "pointer")
          .on("click", (event, d) => options.onClusterClick(d[0], d[1]));
      }

      // Draw cluster labels
      labelLayer.selectAll(".cluster-label")
        .data(clusterEntries)
        .enter()
        .append("text")
        .attr("class", "cluster-label")
        .attr("x", d => d[1].center[0])
        .attr("y", d => d[1].center[1] + Math.sqrt(d[1].size) * 5 + 15)
        .attr("text-anchor", "middle")
        .attr("font-size", "14px")
        .attr("fill", "#333")
        .text(d => d[1].name);

      return {
        linkLayer,
        clusterLayer,
        labelLayer,
        transporterLayer
      };
    },

    _getClusterColor: function(dominantType) {
      // Color scheme for different department types
      const colorMap = {
        "Emergency": "#e74c3c",
        "Surgery": "#9b59b6",
        "Inpatient": "#3498db",
        "Diagnostic": "#2ecc71",
        "Outpatient": "#f39c12",
        "Support": "#7f8c8d",
        "Other": "#95a5a6"
      };

      return colorMap[dominantType] || colorMap.Other;
    },

    getDepartmentsInCluster: function(clusterId) {
      if (!HospitalTransport.clusters.data || !HospitalTransport.clusters.data.clusters) {
        return [];
      }

      const cluster = HospitalTransport.clusters.data.clusters[clusterId];
      return cluster ? cluster.departments : [];
    },

    renderClusterDetails: function(clusterId, svgSelector, options = {}) {
      const departments = this.getDepartmentsInCluster(clusterId);
      if (!departments || departments.length === 0) {
        console.error(`No departments found in cluster ${clusterId}`);
        return null;
      }

      // Create a subgraph with only the departments in this cluster
      const subgraph = {
        nodes: HospitalTransport.state.graph.nodes.filter(n => departments.includes(n.id)),
        edges: HospitalTransport.state.graph.edges.filter(e =>
          departments.includes(e.source) && departments.includes(e.target)
        )
      };

      // Store the original graph temporarily
      const originalGraph = HospitalTransport.state.graph;
      HospitalTransport.state.graph = subgraph;

      // Render the subgraph
      const layers = HospitalTransport.graph.render(svgSelector, {
        ...options,
        clear: true
      });

      // Restore the original graph
      HospitalTransport.state.graph = originalGraph;

      return layers;
    },

    enterCluster: function(clusterId, clusterData) {
      // Update state
      HospitalTransport.state.currentCluster = clusterId;

      // Log the action
      HospitalTransport.log.add(`Viewing cluster: ${clusterData.name}`, "info");

      // Render departments in this cluster
      this.renderClusterDetails(clusterId, 'svg', {
        shortLabels: false,
        showEdgeWeights: true
      });

      // Add back button if it doesn't exist
      if (!document.getElementById('back-to-clusters-btn')) {
        const svg = d3.select('svg');
        const backG = svg.append('g')
          .attr('id', 'back-button-group')
          .attr('transform', 'translate(20, 60)') // Position below clock if it exists
          .style('cursor', 'pointer')
          .on('click', () => HospitalTransport.clusters.exitCluster());

        backG.append('rect')
          .attr('id', 'back-to-clusters-btn')
          .attr('width', 120)
          .attr('height', 30)
          .attr('rx', 5)
          .attr('fill', 'rgba(44, 62, 80, 0.8)');

        backG.append('text')
          .attr('x', 60)
          .attr('y', 20)
          .attr('text-anchor', 'middle')
          .attr('fill', 'white')
          .attr('font-size', '12px')
          .text('◀ Back to Clusters');
      }

      // Update transporters to show only those in this cluster
      const departments = this.getDepartmentsInCluster(clusterId);
      HospitalTransport.transporters.load(() => {
        HospitalTransport.transporters.render('svg', {
          filterDepartments: departments
        });
      });
    },

    exitCluster: function() {
      // Reset current cluster in state
      HospitalTransport.state.currentCluster = null;

      // Remove back button
      d3.select('#back-button-group').remove();

      // Return to cluster view
      this.renderClusterView('svg', {
        clear: true,
        onClusterClick: (clusterId, clusterData) => {
          this.enterCluster(clusterId, clusterData);
        }
      });

      // Update transporters to position by cluster
      HospitalTransport.transporters.load(() => {
        HospitalTransport.transporters.render('svg', {
          positionByCluster: true
        });
      });

      // Log the action
      HospitalTransport.log.add("Returned to cluster overview", "info");
    }
  },

  /**
   * Socket event listeners setup
   */
  _setupSocketListeners: function() {
    const socket = HospitalTransport.state.socket;
    if (!socket) {
      console.error("Socket not initialized");
      return;
    }

    // Transport log events
    socket.on("transport_log", function(data) {
      HospitalTransport.log.add(data.message);
    });

    // Clock tick events
    socket.on("clock_tick", function(data) {
      HospitalTransport.ui.updateClock(data.simTime);
    });

    // Transporter update events
    socket.on("transporter_update", function(data) {
      console.log("Transporter update received:", data);

      // Skip if missing data
      if (!data || !data.name || !data.path || data.path.length < 2) {
        console.warn("Invalid transporter update data", data);
        return;
      }

      // Handle different view modes
      if (HospitalTransport.state.currentView === "clustered") {
        // In cluster view mode
        if (HospitalTransport.state.currentCluster) {
          // In a specific cluster - only show relevant movements
          const departments = HospitalTransport.clusters.getDepartmentsInCluster(
            HospitalTransport.state.currentCluster
          );

          // Filter path to only include departments in this cluster
          const relevantSegments = getRelevantPathSegments(data.path, departments, data.durations);

          if (relevantSegments.path.length >= 2) {
            HospitalTransport.transporters.animatePath(
              data.name,
              relevantSegments.path,
              relevantSegments.durations
            );
          }
        } else {
          // In cluster overview - animate between clusters
          const clusterPath = convertDepartmentPathToClusterPath(
            data.name,
            data.path,
            data.durations
          );

          if (clusterPath.path.length >= 2) {
            HospitalTransport.transporters.animatePath(
              data.name,
              clusterPath.path,
              clusterPath.durations
            );
          }
        }
      } else {
        // In detailed view - regular animation
        HospitalTransport.transporters.animatePath(data.name, data.path, data.durations);
      }
    });

    // New transporter events
    socket.on("new_transporter", function(data) {
      console.log("New transporter added:", data);
      HospitalTransport.state.transporters[data.name] = data;

      // Refresh transporter visualization
      HospitalTransport.transporters.render('svg');
    });

    // Transport status update events
    socket.on("transport_status_update", function(data) {
      console.log("Transport status update:", data);
      HospitalTransport.requests.load();
      HospitalTransport.requests.updateTransportTable();
    });

    // Transport completed events
    socket.on("transport_completed", function(data) {
      console.log("Transport completed:", data);
      HospitalTransport.requests.load();
      HospitalTransport.requests.updateTransportTable();
    });

    // Connection events
    socket.on("connect", function() {
      console.log("Connected to server");
      HospitalTransport.log.add("Connected to server", "success");
    });

    socket.on("disconnect", function() {
      console.log("Disconnected from server");
      HospitalTransport.log.add("Disconnected from server", "error");
    });
  },

  /**
   * Initialization
   */
  initialize: function(options = {}) {
    // Apply configuration
    if (options.config) {
      HospitalTransport.config = {...HospitalTransport.config, ...options.config};
    }

    // Initialize socket
    HospitalTransport.socket.initialize(options.socketUrl);

    // Load graph
    HospitalTransport.graph.load(() => {
      console.log("Hospital graph loaded");

      // Load transporters
      HospitalTransport.transporters.load(() => {
        console.log("Transporters loaded");
      });

      // Load transport requests
      HospitalTransport.requests.load(() => {
        console.log("Transport requests loaded");
      });

      // Try to load clusters if endpoint exists
      fetch(`${HospitalTransport.config.socketUrl}/get_hospital_clusters`)
        .then(response => {
          if (response.ok) {
            return response.json();
          }
          throw new Error("Clusters endpoint not available");
        })
        .then(clusters => {
          HospitalTransport.clusters.data = clusters;
          console.log("Hospital clusters loaded");
        })
        .catch(error => {
          console.log("Clusters not available:", error.message);
        });
    });

    return HospitalTransport;
  }
};

/**
 * Helper function to convert department path to cluster path
 * @param {string} transporterName - Name of the transporter
 * @param {array} departmentPath - Array of department IDs
 * @param {array} durations - Array of durations for each path segment
 * @returns {object} Object with path and durations arrays
 */
function convertDepartmentPathToClusterPath(transporterName, departmentPath, durations) {
  const clusterPath = [];
  const clusterDurations = [];
  let currentCluster = null;
  let accumulatedDuration = 0;

  // Map each department to its cluster
  for (let i = 0; i < departmentPath.length; i++) {
    const dept = departmentPath[i];
    const cluster = HospitalTransport.clusters.getClusterForDepartment(dept);

    // If we've moved to a new cluster
    if (cluster !== currentCluster && cluster) {
      if (currentCluster !== null) {
        // Add the previous cluster to the path
        clusterPath.push(currentCluster);
        clusterDurations.push(accumulatedDuration);
        accumulatedDuration = 0;
      }
      currentCluster = cluster;
    }

    // Add this step's duration
    if (i < departmentPath.length - 1 && i < durations.length) {
      accumulatedDuration += durations[i];
    }
  }

  // Add the final cluster if it exists
  if (currentCluster && (clusterPath.length === 0 || clusterPath[clusterPath.length - 1] !== currentCluster)) {
    clusterPath.push(currentCluster);
    clusterDurations.push(accumulatedDuration);
  }

  return {
    path: clusterPath,
    durations: clusterDurations
  };
}

/**
 * Helper function to get relevant segments of a path that include specified departments
 * @param {array} path - Array of department IDs
 * @param {array} departments - Array of department IDs to filter by
 * @param {array} durations - Array of durations for each path segment
 * @returns {object} Object with filtered path and durations arrays
 */
function getRelevantPathSegments(path, departments, durations) {
  const filteredPath = [];
  const filteredDurations = [];

  for (let i = 0; i < path.length; i++) {
    if (departments.includes(path[i])) {
      filteredPath.push(path[i]);

      // Add duration if there is a next segment
      if (i < path.length - 1 && i < durations.length) {
        filteredDurations.push(durations[i]);
      }
    }
  }

  return {
    path: filteredPath,
    durations: filteredDurations
  };
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = HospitalTransport;
}