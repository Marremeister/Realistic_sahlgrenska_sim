/**
 * clustered-view.js - Hierarchical cluster visualization for hospital transport system
 * Utilizes hospital-transport-utils.js library
 */

// Initialize the HospitalTransport library
const transportSystem = HospitalTransport.initialize();

// Global state for cluster view
const clusterView = {
  currentView: "clusters", // "clusters" or "departments"
  currentCluster: null
};

// Initialize when document is ready
window.onload = function() {
  // Load hospital graph and clusters
  transportSystem.graph.load(() => {
    // Load clusters
    transportSystem.clusters.load((clusterData) => {
      // Initialize the view
      initializeClusterView();
    });
  });

  // Set up UI elements
  setupUIElements();
};

function initializeClusterView() {
  // Render the cluster view
  renderClusters();

  // Set up refresh interval for transporters
  setInterval(() => {
    updateTransporters();
  }, 5000);
}

function renderClusters() {
  // Render clusters using the transportSystem
  transportSystem.clusters.renderClusterView('svg', {
    clear: true,
    onClusterClick: enterCluster
  });

  // Update transporters to show in clusters
  updateTransporters();
}

function enterCluster(clusterId, clusterData) {
  // Update state
  clusterView.currentView = "departments";
  clusterView.currentCluster = clusterId;

  // Show back button
  document.getElementById("back-to-clusters-btn").style.display = "block";

  // Update view indicator
  document.getElementById("current-view-indicator").textContent = `Viewing: ${clusterData.name}`;

  // Render department details for this cluster
  transportSystem.clusters.renderClusterDetails(clusterId, 'svg', {
    shortLabels: false,
    showEdgeWeights: true
  });

  // Update transporters to show only those in this cluster
  updateTransporters();
}

function exitCluster() {
  // Update state
  clusterView.currentView = "clusters";
  clusterView.currentCluster = null;

  // Hide back button
  document.getElementById("back-to-clusters-btn").style.display = "none";

  // Update view indicator
  document.getElementById("current-view-indicator").textContent = "Viewing: Hospital Clusters";

  // Return to cluster view
  renderClusters();
}

function updateTransporters() {
  // Load transporter data
  transportSystem.transporters.load(() => {
    // If viewing clusters, position transporters at cluster centers
    if (clusterView.currentView === "clusters") {
      // Create a map of which cluster each transporter is in
      const transporterClusters = {};

      // For each transporter, determine its cluster
      Object.values(transportSystem.state.transporters).forEach(transporter => {
        const departmentCluster = transportSystem.clusters.getClusterForDepartment(transporter.current_location);
        if (departmentCluster) {
          transporterClusters[transporter.name] = departmentCluster;
        }
      });

      // Update the visualization to show transporters at cluster centers
      transportSystem.transporters.render('svg', {
        positionByCluster: true,
        transporterClusters: transporterClusters
      });
    }
    // If viewing departments within a cluster, show only transporters in that cluster
    else if (clusterView.currentView === "departments" && clusterView.currentCluster) {
      // Get departments in this cluster
      const departments = transportSystem.clusters.getDepartmentsInCluster(clusterView.currentCluster);

      // Render only transporters in these departments
      transportSystem.transporters.render('svg', {
        filterDepartments: departments
      });
    }
  });
}

function setupUIElements() {
  // Add back button to controls
  const controlsDiv = document.getElementById("visualization-controls");

  // Add back button
  const backButton = document.createElement("button");
  backButton.id = "back-to-clusters-btn";
  backButton.className = "btn";
  backButton.innerHTML = "◀ Back to Clusters";
  backButton.style.display = "none";
  backButton.addEventListener("click", exitCluster);
  controlsDiv.appendChild(backButton);

  // Add view indicator
  const viewIndicator = document.createElement("div");
  viewIndicator.id = "current-view-indicator";
  viewIndicator.className = "view-indicator";
  viewIndicator.textContent = "Viewing: Hospital Clusters";
  controlsDiv.appendChild(viewIndicator);

  // Setup apply clustering button
  document.getElementById("applyClusteringBtn").addEventListener("click", applyClusterMethod);
}

applyClusterMethod: function(method) {
    // Show loading message
    HospitalTransport.log.add(`Applying ${method} clustering...`, "info");

    // Call backend to apply clustering
    fetch("/apply_clustering", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method: method })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            HospitalTransport.log.add(`Error: ${data.error}`, "error");
        } else {
            HospitalTransport.log.add(data.message || "Clustering applied successfully", "success");

            // Update UI elements if they exist
            const currentMethodEl = document.getElementById('currentClusterMethod');
            if (currentMethodEl) currentMethodEl.textContent = method;

            // Load clusters and render
            HospitalTransport.clusters.load((clusterData) => {
                // Make sure we have the graph loaded
                if (!HospitalTransport.state.graph) {
                    HospitalTransport.graph.load(() => {
                        renderClusters(clusterData);
                    });
                } else {
                    renderClusters(clusterData);
                }
            });
        }
    })
    .catch(error => {
        console.error('Error applying clustering method:', error);
        HospitalTransport.log.add(`Failed to apply clustering: ${error}`, "error");

        // Fall back to detailed view
        const viewModeSelect = document.getElementById('viewModeSelect');
        if (viewModeSelect) {
            viewModeSelect.value = 'detailed';
            this.changeViewMode('detailed');
        }
    });

    // Helper function to render clusters
    function renderClusters(clusterData) {
        // Update cluster count if element exists
        const clusterCountEl = document.getElementById('clusterCount');
        if (clusterCountEl && clusterData && clusterData.clusters) {
            clusterCountEl.textContent = Object.keys(clusterData.clusters).length;
        }

        // Render cluster view
        HospitalTransport.clusters.renderClusterView('svg', {
            clear: true,
            onClusterClick: (clusterId, clusterData) => {
                enterCluster(clusterId, clusterData);
            }
        });

        // Load transporters and position them by cluster
        HospitalTransport.transporters.load(() => {
            HospitalTransport.transporters.render('svg', {
                positionByCluster: true
            });
        });
    }
}

// Handle transporter animations at both cluster and department level
transportSystem.socket.onEvent("transporter_update", function(data) {
  if (!data || !data.name || !data.path || data.path.length < 2) return;

  // Custom handling based on current view
  if (clusterView.currentView === "clusters") {
    // In cluster view, convert department path to cluster path
    const clusterPath = convertDepartmentPathToClusterPath(data.name, data.path, data.durations);

    // Only animate if we have a valid cluster path
    if (clusterPath.path.length >= 2) {
      // Get the transporter elements
      const svg = d3.select('svg');
      const transporterLayer = svg.select(".transporter-layer");

      // Animate at the cluster level
      transportSystem.transporters.animatePath(
        data.name,
        clusterPath.path,
        clusterPath.durations,
        'svg'
      );
    }
  }
  else if (clusterView.currentView === "departments" && clusterView.currentCluster) {
    // Only animate if the transporter's path includes departments in this cluster
    const departments = transportSystem.clusters.getDepartmentsInCluster(clusterView.currentCluster);

    // Filter path to only include departments in this cluster
    const relevantSegments = getRelevantPathSegments(data.path, departments);

    if (relevantSegments.length >= 2) {
      // Animate the transporter within the cluster
      transportSystem.transporters.animatePath(
        data.name,
        relevantSegments.map(s => s.department),
        relevantSegments.map(s => s.duration),
        'svg'
      );
    }
  }
});

// Helper function to convert department path to cluster path
function convertDepartmentPathToClusterPath(transporterName, departmentPath, durations) {
  const clusterPath = [];
  const clusterDurations = [];
  let currentCluster = null;
  let accumulatedDuration = 0;

  // Map each department to its cluster
  for (let i = 0; i < departmentPath.length; i++) {
    const dept = departmentPath[i];
    const cluster = transportSystem.clusters.getClusterForDepartment(dept);

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

// Helper function to get relevant segments of a path that include specified departments
function getRelevantPathSegments(path, departments) {
  const segments = [];

  for (let i = 0; i < path.length; i++) {
    if (departments.includes(path[i])) {
      segments.push({
        department: path[i],
        duration: (i < path.length - 1) ? 1000 : 0 // Default duration
      });
    }
  }

  return segments;
}