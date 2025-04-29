/**
 * transport-visualization.js
 * Shared library for hospital transport visualization components
 * Used by both the benchmark and simulation systems
 */

// Namespace for visualization components
const TransportVisualization = {
  /**
   * Configuration
   */
  config: {
    colors: {
      // Strategy colors for consistent visualization
      'ILP: Makespan': 'rgba(75, 108, 183, 0.7)',
      'ILP: Equal Workload': 'rgba(241, 196, 15, 0.7)',
      'ILP: Urgency First': 'rgba(230, 126, 34, 0.7)',
      'ILP: Cluster-Based': 'rgba(46, 204, 113, 0.7)',
      'Genetic Algorithm': 'rgba(155, 89, 182, 0.7)',
      'Random': 'rgba(231, 76, 60, 0.7)',
      'Simulation': 'rgba(52, 152, 219, 0.7)'
    },
    borderColors: {
      'ILP: Makespan': 'rgba(75, 108, 183, 1)',
      'ILP: Equal Workload': 'rgba(241, 196, 15, 1)',
      'ILP: Urgency First': 'rgba(230, 126, 34, 1)',
      'ILP: Cluster-Based': 'rgba(46, 204, 113, 1)',
      'Genetic Algorithm': 'rgba(155, 89, 182, 1)',
      'Random': 'rgba(231, 76, 60, 1)',
      'Simulation': 'rgba(52, 152, 219, 1)'
    }
  },

  /**
   * Initialize the visualization library
   * @param {Object} options - Configuration options
   */
  initialize: function(options = {}) {
    // Override default configuration
    if (options.colors) {
      this.config.colors = {...this.config.colors, ...options.colors};
    }
    if (options.borderColors) {
      this.config.borderColors = {...this.config.borderColors, ...options.borderColors};
    }

    console.log("Transport visualization library initialized");
    return this;
  },

  /**
   * Create summary result cards for key metrics
   * @param {string} containerId - DOM container to create cards in
   * @param {Object} results - Results data with optimal, random, improvement and std values
   */
  createSummaryCards: function(containerId, results) {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container ${containerId} not found`);
      return;
    }

    container.innerHTML = `
      <div class="result-cards">
        <div class="result-card">
          <div class="card-title">Optimal Makespan</div>
          <div class="card-value" id="optimal-makespan">${results.optimal.toFixed(2)}</div>
          <div class="card-unit">seconds</div>
        </div>
        <div class="result-card">
          <div class="card-title">Random Average</div>
          <div class="card-value" id="random-average">${results.random.toFixed(2)}</div>
          <div class="card-unit">seconds</div>
        </div>
        <div class="result-card highlight">
          <div class="card-title">Improvement</div>
          <div class="card-value" id="improvement-percentage">${results.improvement.toFixed(1)}</div>
          <div class="card-unit">percent</div>
        </div>
        <div class="result-card">
          <div class="card-title">Std. Deviation</div>
          <div class="card-value" id="random-std">${results.std.toFixed(2)}</div>
          <div class="card-unit">seconds</div>
        </div>
      </div>
    `;
  },

  /**
   * Create a metrics comparison chart
   * @param {string} canvasId - Canvas element ID
   * @param {Array} strategies - Array of strategy results to display
   * @returns {Object} Created Chart.js instance
   */
  createMetricsChart: function(canvasId, strategies) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // Process strategy data
    const datasets = strategies.map(strategy => {
      const color = this.config.colors[strategy.name] || this.config.colors['Simulation'];
      const borderColor = this.config.borderColors[strategy.name] || this.config.borderColors['Simulation'];

      return {
        label: strategy.name,
        backgroundColor: color,
        borderColor: borderColor,
        borderWidth: 1,
        data: [
          strategy.metrics.mean,
          strategy.metrics.median,
          strategy.metrics.std,
          strategy.metrics.min,
          strategy.metrics.max
        ]
      };
    });

    // Create chart
    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Comparison of Key Metrics'
          },
          legend: {
            position: 'top',
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Time (seconds)'
            }
          }
        }
      }
    });

    return chart;
  },

  /**
   * Create a histogram chart for completion times
   * @param {string} canvasId - Canvas element ID
   * @param {Array} times - Array of completion times
   * @param {number} optimalTime - The optimal completion time
   * @returns {Object} Created Chart.js instance
   */
  createHistogramChart: function(canvasId, times, optimalTime) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // Calculate histogram bins
    const min = Math.min(...times);
    const max = Math.max(...times);
    const range = max - min;
    const binCount = 10;
    const binWidth = range / binCount;

    // Create bins
    const bins = Array(binCount).fill(0);

    // Count values in each bin
    times.forEach(time => {
      const binIndex = Math.min(Math.floor((time - min) / binWidth), binCount - 1);
      bins[binIndex]++;
    });

    // Create labels
    const labels = [];
    for (let i = 0; i < binCount; i++) {
      const start = min + (i * binWidth);
      const end = min + ((i + 1) * binWidth);
      labels.push(`${start.toFixed(1)}-${end.toFixed(1)}`);
    }

    // Calculate mean for annotation
    const mean = times.reduce((sum, val) => sum + val, 0) / times.length;

    // Create chart
    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Frequency',
          backgroundColor: this.config.colors['Random'],
          borderColor: this.config.borderColors['Random'],
          borderWidth: 1,
          data: bins
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Distribution of Completion Times'
          },
          legend: {
            position: 'top',
          },
          annotation: {
            annotations: {
              optimalLine: {
                type: 'line',
                xMin: optimalTime,
                xMax: optimalTime,
                borderColor: this.config.borderColors['ILP: Makespan'],
                borderWidth: 2,
                label: {
                  content: `Optimal: ${optimalTime.toFixed(2)}s`,
                  enabled: true
                }
              },
              meanLine: {
                type: 'line',
                xMin: mean,
                xMax: mean,
                borderColor: this.config.borderColors['Random'],
                borderWidth: 2,
                label: {
                  content: `Mean: ${mean.toFixed(2)}s`,
                  enabled: true
                }
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Frequency'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Completion Time (seconds)'
            }
          }
        }
      }
    });

    return chart;
  },

  /**
   * Create a workload distribution chart
   * @param {string} canvasId - Canvas element ID
   * @param {string} title - Chart title
   * @param {Object} workload - Workload data (transporter->time)
   * @param {string} strategyName - Name of strategy (for coloring)
   * @returns {Object} Created Chart.js instance
   */
  createWorkloadChart: function(canvasId, title, workload, strategyName) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // Extract transporters and workloads
    const transporters = Object.keys(workload);
    const workloads = transporters.map(t => workload[t]);

    // Calculate standard deviation
    const mean = workloads.reduce((sum, val) => sum + val, 0) / workloads.length;
    const variance = workloads.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / workloads.length;
    const std = Math.sqrt(variance);

    // Get colors
    const color = this.config.colors[strategyName] || this.config.colors['Simulation'];
    const borderColor = this.config.borderColors[strategyName] || this.config.borderColors['Simulation'];

    // Create chart
    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: transporters,
        datasets: [{
          label: 'Total Work Time',
          backgroundColor: color,
          borderColor: borderColor,
          borderWidth: 1,
          data: workloads
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: title
          },
          subtitle: {
            display: true,
            text: `Standard Deviation: ${std.toFixed(2)}`
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Total Time (seconds)'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Transporter'
            }
          }
        }
      }
    });

    return chart;
  },

  /**
   * Create or update a comparison table
   * @param {string} tableId - Table element ID
   * @param {Array} strategies - Array of strategy results
   */
  updateComparisonTable: function(tableId, strategies) {
    const tableBody = document.querySelector(`#${tableId} tbody`);
    if (!tableBody) {
      console.error(`Table body for ${tableId} not found`);
      return;
    }

    // Clear existing rows
    tableBody.innerHTML = '';

    // Find best mean time among non-random strategies
    const bestMean = Math.min(...strategies
      .filter(s => s.name !== 'Random')
      .map(s => s.metrics.mean));

    // Add a row for each strategy
    strategies.forEach(strategy => {
      const row = document.createElement('tr');

      row.innerHTML = `
        <td>${strategy.name}</td>
        <td>${strategy.metrics.mean.toFixed(2)}s</td>
        <td>${strategy.metrics.median.toFixed(2)}s</td>
        <td>${strategy.metrics.std.toFixed(2)}</td>
        <td>${strategy.metrics.max.toFixed(2)}s</td>
        <td>${strategy.metrics.workloadStd.toFixed(2)}</td>
      `;

      // Highlight the best mean time
      if (strategy.name !== 'Random' && strategy.metrics.mean === bestMean) {
        row.querySelector('td:nth-child(2)').style.fontWeight = 'bold';
        row.querySelector('td:nth-child(2)').style.color = '#27ae60';
      }

      tableBody.appendChild(row);
    });
  },

  /**
   * Create a time-based parameter card
   * @param {string} containerId - Container element ID
   * @param {number} startHour - Start hour (0-23)
   * @param {number} endHour - End hour (0-23)
   * @param {number} transporterCount - Number of transporters
   * @param {number} requestRate - Average requests per hour
   */
  createTimeParamsCard: function(containerId, startHour, endHour, transporterCount, requestRate) {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container ${containerId} not found`);
      return;
    }

    // Create card
    const card = document.createElement('div');
    card.className = 'time-params-card';

    card.innerHTML = `
      <div class="time-params-title">Time-based Parameters</div>
      <div class="time-param-item">
        <span class="time-param-label">Time Range:</span>
        <span class="time-param-value">${startHour}:00 - ${endHour}:00</span>
      </div>
      <div class="time-param-item">
        <span class="time-param-label">Transporters:</span>
        <span class="time-param-value">${transporterCount}</span>
      </div>
      <div class="time-param-item">
        <span class="time-param-label">Requests/Hour:</span>
        <span class="time-param-value">${typeof requestRate === 'number' ? requestRate.toFixed(2) : requestRate}</span>
      </div>
    `;

    // Add to container (at beginning)
    if (container.firstChild) {
      container.insertBefore(card, container.firstChild);
    } else {
      container.appendChild(card);
    }

    return card;
  },

  /**
   * Create UI for time range selection
   * @param {string} containerId - Container element ID
   * @param {Function} onRangeSelected - Callback when range is selected
   * @param {Array} predefinedRanges - Optional array of predefined time ranges
   */
  createTimeRangeSelector: function(containerId, onRangeSelected, predefinedRanges = []) {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container ${containerId} not found`);
      return;
    }

    // Create time range selector elements
    const selectorDiv = document.createElement('div');
    selectorDiv.className = 'time-range-selector';

    selectorDiv.innerHTML = `
      <div class="form-group">
        <label>Time Range:</label>
        <div class="time-range-selector-controls">
          <select id="start-hour" class="time-select"></select>
          <span class="time-separator">to</span>
          <select id="end-hour" class="time-select"></select>
        </div>
      </div>
    `;

    // Add predefined ranges if provided
    if (predefinedRanges && predefinedRanges.length > 0) {
      const rangesDiv = document.createElement('div');
      rangesDiv.className = 'form-group';
      rangesDiv.innerHTML = `<label>Suggested Time Ranges:</label>`;

      const buttonContainer = document.createElement('div');
      buttonContainer.className = 'suggested-time-ranges';

      predefinedRanges.forEach(range => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn small secondary time-range-btn';
        button.textContent = range;

        button.addEventListener('click', () => {
          const [start, end] = range.split('-').map(h => parseInt(h));

          // Update selects
          const startHourSelect = document.getElementById('start-hour');
          const endHourSelect = document.getElementById('end-hour');

          if (startHourSelect && endHourSelect) {
            startHourSelect.value = start;
            endHourSelect.value = end;

            // Trigger callback
            if (onRangeSelected) {
              onRangeSelected(start, end);
            }
          }

          // Set active state
          document.querySelectorAll('.time-range-btn').forEach(btn => {
            btn.classList.remove('active');
          });
          button.classList.add('active');
        });

        buttonContainer.appendChild(button);
      });

      rangesDiv.appendChild(buttonContainer);
      selectorDiv.appendChild(rangesDiv);
    }

    container.appendChild(selectorDiv);

    // Populate hour options
    const startHourSelect = document.getElementById('start-hour');
    const endHourSelect = document.getElementById('end-hour');

    if (startHourSelect && endHourSelect) {
      // Add hour options (0-23)
      for (let i = 0; i < 24; i++) {
        const formattedHour = i.toString().padStart(2, '0') + ':00';

        const startOption = document.createElement('option');
        startOption.value = i;
        startOption.textContent = formattedHour;
        startHourSelect.appendChild(startOption);

        const endOption = document.createElement('option');
        endOption.value = i;
        endOption.textContent = formattedHour;
        endHourSelect.appendChild(endOption);
      }

      // Set common business hours as default (9am-5pm)
      startHourSelect.value = 9;
      endHourSelect.value = 17;

      // Add change event listeners
      startHourSelect.addEventListener('change', () => {
        if (onRangeSelected) {
          onRangeSelected(parseInt(startHourSelect.value), parseInt(endHourSelect.value));
        }
      });

      endHourSelect.addEventListener('change', () => {
        if (onRangeSelected) {
          onRangeSelected(parseInt(startHourSelect.value), parseInt(endHourSelect.value));
        }
      });
    }

    return selectorDiv;
  },

  /**
   * Create simulation control UI
   * @param {string} containerId - Container element ID
   * @param {Object} options - Options for controls
   * @param {Function} options.onStart - Start callback
   * @param {Function} options.onStop - Stop callback
   * @param {Function} options.onSpeedChange - Speed change callback
   * @returns {Object} References to created UI elements
   */
  createSimulationControls: function(containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container ${containerId} not found`);
      return null;
    }

    // Create control elements
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'simulation-controls';

    // Create speed slider if requested
    let speedSlider = null;
    if (options.showSpeedControl) {
      const speedDiv = document.createElement('div');
      speedDiv.className = 'form-group';
      speedDiv.innerHTML = `
        <label for="simulation-speed">Simulation Speed:</label>
        <div class="slider-container">
          <input type="range" id="simulation-speed" min="1" max="100" value="10" class="slider">
          <span id="speed-value" class="slider-value">10x</span>
        </div>
      `;
      controlsDiv.appendChild(speedDiv);

      // Set up speed slider
      speedSlider = speedDiv.querySelector('#simulation-speed');
      const speedValue = speedDiv.querySelector('#speed-value');

      speedSlider.addEventListener('input', function() {
        const speed = parseInt(this.value);
        speedValue.textContent = `${speed}x`;

        if (options.onSpeedChange) {
          options.onSpeedChange(speed);
        }
      });
    }

    // Create skip option if requested
    let skipCheckbox = null;
    if (options.showSkipOption) {
      const skipDiv = document.createElement('div');
      skipDiv.className = 'form-group';
      skipDiv.innerHTML = `
        <label class="checkbox-label">
          <input type="checkbox" id="skip-simulation">
          <span>Skip to Results</span>
        </label>
      `;
      controlsDiv.appendChild(skipDiv);

      skipCheckbox = skipDiv.querySelector('#skip-simulation');
    }

    // Create start/stop buttons
    const buttonDiv = document.createElement('div');
    buttonDiv.className = 'btn-group';
    buttonDiv.innerHTML = `
      <button id="start-simulation-btn" class="btn success">
        <i class="fas fa-play"></i> Start Simulation
      </button>
      <button id="stop-simulation-btn" class="btn danger" style="display: none;">
        <i class="fas fa-stop"></i> Stop Simulation
      </button>
    `;
    controlsDiv.appendChild(buttonDiv);

    // Set up button event handlers
    const startBtn = buttonDiv.querySelector('#start-simulation-btn');
    const stopBtn = buttonDiv.querySelector('#stop-simulation-btn');

    startBtn.addEventListener('click', function() {
      startBtn.style.display = 'none';
      stopBtn.style.display = 'inline-block';

      if (options.onStart) {
        const skipToResults = skipCheckbox ? skipCheckbox.checked : false;
        const speed = speedSlider ? parseInt(speedSlider.value) : 10;
        options.onStart(skipToResults, speed);
      }
    });

    stopBtn.addEventListener('click', function() {
      stopBtn.style.display = 'none';
      startBtn.style.display = 'inline-block';

      if (options.onStop) {
        options.onStop();
      }
    });

    container.appendChild(controlsDiv);

    // Return references to created elements
    return {
      controls: controlsDiv,
      startButton: startBtn,
      stopButton: stopBtn,
      speedSlider,
      skipCheckbox
    };
  },

  /**
   * Create tabs for displaying different result views
   * @param {string} containerId - Container element ID
   * @param {Array} tabs - Array of tab configurations
   * @returns {Object} References to created tab elements
   */
  createResultTabs: function(containerId, tabs) {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container ${containerId} not found`);
      return null;
    }

    // Create tab buttons
    const tabButtons = document.createElement('div');
    tabButtons.className = 'benchmark-tabs';

    // Create tab content container
    const tabContent = document.createElement('div');
    tabContent.className = 'tab-content';

    // Create each tab
    tabs.forEach((tab, index) => {
      // Create tab button
      const button = document.createElement('button');
      button.className = `tab-btn ${index === 0 ? 'active' : ''}`;
      button.setAttribute('data-tab', `tab-${tab.id}`);
      button.textContent = tab.title;
      tabButtons.appendChild(button);

      // Create tab pane
      const pane = document.createElement('div');
      pane.id = `tab-${tab.id}`;
      pane.className = `tab-pane ${index === 0 ? 'active' : ''}`;

      // If content creator function is provided, call it
      if (tab.createContent) {
        tab.createContent(pane);
      }

      tabContent.appendChild(pane);
    });

    // Add event listeners to tab buttons
    tabButtons.addEventListener('click', function(event) {
      if (event.target.classList.contains('tab-btn')) {
        // Remove active class from all buttons and panes
        tabButtons.querySelectorAll('.tab-btn').forEach(btn => {
          btn.classList.remove('active');
        });
        tabContent.querySelectorAll('.tab-pane').forEach(pane => {
          pane.classList.remove('active');
        });

        // Add active class to clicked button and corresponding pane
        event.target.classList.add('active');
        const tabId = event.target.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
      }
    });

    // Add to container
    container.appendChild(tabButtons);
    container.appendChild(tabContent);

    return {
      tabButtons,
      tabContent,
      tabPanes: tabs.map(tab => document.getElementById(`tab-${tab.id}`))
    };
  },

  /**
   * Utility function to calculate statistics for an array of values
   * @param {Array} values - Array of numeric values
   * @returns {Object} Object with calculated statistics
   */
  calculateStatistics: function(values) {
    if (!values || values.length === 0) {
      return {
        mean: 0,
        median: 0,
        std: 0,
        min: 0,
        max: 0,
        count: 0
      };
    }

    // Calculate mean
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;

    // Calculate median
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];

    // Calculate standard deviation
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);

    return {
      mean,
      median,
      std,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length
    };
  },

  /**
   * Format simulation results for visualization
   * @param {Object} simulationData - Raw simulation results
   * @returns {Object} Formatted data ready for visualization
   */
  formatSimulationResults: function(simulationData) {
    // Extract key data
    const completionTimes = simulationData.completionTimes || [];
    const optimalTime = simulationData.optimalTime || 0;
    const workloads = simulationData.workloads || {};

    // Calculate statistics
    const timeStats = this.calculateStatistics(completionTimes);
    const workloadValues = Object.values(workloads.optimal || {});
    const workloadStats = this.calculateStatistics(workloadValues);

    // Calculate improvement percentage
    const improvement = ((timeStats.mean - optimalTime) / timeStats.mean) * 100;

    // Format strategies for charts
    const strategies = [
      {
        name: simulationData.strategyName || 'Simulation',
        times: [optimalTime],
        workload: workloads.optimal || {},
        metrics: {
          mean: optimalTime,
          median: optimalTime,
          std: 0,
          min: optimalTime,
          max: optimalTime,
          workloadStd: workloadStats.std
        }
      },
      {
        name: 'Random',
        times: completionTimes,
        workload: workloads.random || {},
        metrics: {
          ...timeStats,
          workloadStd: this.calculateStatistics(Object.values(workloads.random || {})).std
        }
      }
    ];

    // Return formatted results
    return {
      optimal: optimalTime,
      random: timeStats.mean,
      improvement: improvement,
      std: timeStats.std,
      strategies,
      timeRange: {
        start: simulationData.startHour || 9,
        end: simulationData.endHour || 17,
        transporters: simulationData.transporterCount || 3,
        requestRate: simulationData.requestRate || 0
      }
    };
  }
};

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TransportVisualization;
}