/**
 * Incremental benchmark extensions for the hospital transport benchmark system.
 * Adds time-based request simulation and visualization.
 */

// Global state for incremental benchmarks
let incrementalResults = {
    strategies: {},
    hasData: false
};

// DOM Elements for incremental benchmark UI
let incrementalToggle;

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM loaded, initializing incremental benchmark...");
    initializeIncrementalBenchmark();
    setupIncrementalSocketListeners();
    loadStylesheet('static/css/benchmark-incremental.css'); // Remove the leading slash
    loadHourlyRateData();
});

// Load incremental benchmark stylesheet
function loadStylesheet(path) {
    console.log("Loading stylesheet:", path);
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.type = 'text/css';
    link.href = path;
    document.head.appendChild(link);
}

// Initialize incremental benchmark components
function initializeIncrementalBenchmark() {
    addIncrementalModeToggle();
    addIncrementalConfigPanel();
    addIncrementalResultsTab();
    setupIncrementalEventListeners();
}

// Add toggle switch for incremental mode
function addIncrementalModeToggle() {
    const configPanel = document.querySelector('.benchmark-container .panel:first-child');
    if (!configPanel) return;

    const toggleHTML = `
    <div class="form-group incremental-toggle-container">
        <label>Execution Mode:</label>
        <div class="benchmark-switch-container">
            <label class="benchmark-type-switch">
                <input type="checkbox" id="incremental-mode-toggle">
                <div class="benchmark-switch-slider"></div>
                <div class="benchmark-switch-labels">
                    <span class="benchmark-switch-label-standard">Standard</span>
                    <span class="benchmark-switch-label-time">Incremental</span>
                </div>
            </label>
        </div>
    </div>
    `;

    const title = configPanel.querySelector('h3');
    if (title) {
        title.insertAdjacentHTML('afterend', toggleHTML);
    } else {
        configPanel.insertAdjacentHTML('afterbegin', toggleHTML);
    }

    incrementalToggle = document.getElementById('incremental-mode-toggle');
}

// Add configuration panel for incremental mode
function addIncrementalConfigPanel() {
    const configHTML = `
    <div id="incremental-config" class="panel" style="display: none;">
        <h3><i class="fas fa-clock"></i> Incremental Mode Settings</h3>

        <div class="form-group">
            <label for="timeRangeSlider">Simulation Duration (minutes):</label>
            <div class="slider-container">
                <input type="range" id="timeRangeSlider" min="5" max="120" value="60" class="slider">
                <span id="timeRangeValue" class="slider-value">60</span>
            </div>
        </div>

        <div class="form-group">
            <label>Time Distribution:</label>
            <div class="time-distribution-options">
                <label class="radio-label">
                    <input type="radio" name="timeDistribution" value="realistic" checked>
                    <span>Realistic (based on hospital data)</span>
                </label>
                <label class="radio-label">
                    <input type="radio" name="timeDistribution" value="uniform">
                    <span>Uniform (evenly spaced)</span>
                </label>
                <label class="radio-label">
                    <input type="radio" name="timeDistribution" value="random">
                    <span>Random</span>
                </label>
            </div>
        </div>

        <div class="form-group">
            <label>Hourly Request Rate:</label>
            <div id="hourly-rate-container" class="hourly-rate-container">
                <canvas id="hourly-rate-chart" height="180"></canvas>
            </div>
            <span class="small-note">Based on actual hospital data</span>
        </div>
    </div>
    `;

    const firstPanel = document.querySelector('.benchmark-container .panel:first-child');
    if (firstPanel) {
        firstPanel.insertAdjacentHTML('afterend', configHTML);
    }

    initializeTimeSlider();
}

// Initialize time range slider
function initializeTimeSlider() {
    const timeRangeSlider = document.getElementById('timeRangeSlider');
    const timeRangeValue = document.getElementById('timeRangeValue');

    if (timeRangeSlider && timeRangeValue) {
        timeRangeSlider.addEventListener('input', function() {
            timeRangeValue.textContent = this.value;
        });
    }
}

// Add tab for timeline visualization
function addIncrementalResultsTab() {
    addTimelineTab();
    addTimelineContent();
    reinitializeTabs();
}

// Add timeline tab button
function addTimelineTab() {
    const tabsContainer = document.querySelector('.benchmark-tabs');
    if (!tabsContainer) return;

    const timelineTab = document.createElement('button');
    timelineTab.className = 'tab-btn';
    timelineTab.setAttribute('data-tab', 'tab-timeline');
    timelineTab.textContent = 'Timeline';
    tabsContainer.appendChild(timelineTab);
}

// Add timeline tab content
function addTimelineContent() {
    const tabContent = document.querySelector('.tab-content');
    if (!tabContent) return;

    const timelinePane = document.createElement('div');
    timelinePane.id = 'tab-timeline';
    timelinePane.className = 'tab-pane';

    timelinePane.innerHTML = `
        <div class="time-results-header">
            <h3>Request Timeline & Performance</h3>
            <div class="time-controls">
                <button id="zoom-all-btn" class="btn small secondary">Show All</button>
                <button id="zoom-peak-btn" class="btn small secondary">Show Peak</button>
            </div>
        </div>

        <div class="timeline-charts">
            <div class="chart-container timeline-chart-container">
                <h4>Request Timeline</h4>
                <canvas id="request-timeline-chart"></canvas>
            </div>

            <div class="chart-container load-chart-container">
                <h4>System Load</h4>
                <canvas id="system-load-chart"></canvas>
            </div>
        </div>

        <div class="chart-container performance-comparison-container">
            <h4>Strategy Performance Over Time</h4>
            <canvas id="strategy-comparison-chart"></canvas>
        </div>

        <div class="chart-description">
            <p>These charts show how each optimization strategy performs in a time-based simulation:</p>
            <ul>
                <li><strong>Request Timeline</strong>: When requests appear in the system</li>
                <li><strong>System Load</strong>: How many active transporters and requests exist at each point in time</li>
                <li><strong>Strategy Performance</strong>: How the makespan estimate evolves as new requests arrive</li>
            </ul>
            <p>This provides a more realistic view of how different strategies would perform in a real hospital environment with varying demand.</p>
        </div>
    `;

    tabContent.appendChild(timelinePane);
}

// Re-initialize tabs after adding new tab
function reinitializeTabs() {
    if (typeof initializeTabs === 'function') {
        initializeTabs();
    }
}

// Set up event listeners for incremental UI
function setupIncrementalEventListeners() {
    setupModeToggle();
    setupZoomButtons();
}

// Handle mode toggle switching
function setupModeToggle() {
    if (!incrementalToggle) return;

    incrementalToggle.addEventListener('change', function() {
        const incrementalConfig = document.getElementById('incremental-config');

        if (this.checked) {
            // Switch to incremental mode
            if (incrementalConfig) incrementalConfig.style.display = 'block';
        } else {
            // Switch to standard mode
            if (incrementalConfig) incrementalConfig.style.display = 'none';
        }
    });
}

// Set up zoom control buttons
function setupZoomButtons() {
    document.getElementById('zoom-all-btn')?.addEventListener('click', function() {
        zoomTimelineCharts('all');
    });

    document.getElementById('zoom-peak-btn')?.addEventListener('click', function() {
        zoomTimelineCharts('peak');
    });
}

// Set up socket listeners for incremental results
function setupIncrementalSocketListeners() {
    socket.on('incremental_benchmark_results', function(data) {
        processIncrementalResults(data);
    });
}

// Process incremental benchmark results from server
function processIncrementalResults(data) {
    const { strategy, time_metrics, events, makespan, workload, simulation_time, hourly_distribution } = data;

    // Store the incremental results
    incrementalResults.strategies[strategy] = {
        time_metrics,
        events,
        makespan,
        workload,
        simulation_time,
        hourly_distribution
    };

    incrementalResults.hasData = true;

    // Update the standard results for compatibility
    processBenchmarkResults({
        strategy,
        times: [makespan],
        workload
    });

    // Update visualizations
    updateIncrementalVisualizations();
}

// Update all incremental visualizations
function updateIncrementalVisualizations() {
    if (!incrementalResults.hasData) return;

    updateRequestTimelineChart();
    updateSystemLoadChart();
    updateStrategyComparisonChart();
}

// Update request timeline chart
function updateRequestTimelineChart() {
    const ctx = document.getElementById('request-timeline-chart')?.getContext('2d');
    if (!ctx) return;

    const allEvents = getRequestEvents();

    // If no events, show empty chart
    if (allEvents.length === 0) {
        createEmptyTimelineChart(ctx);
        return;
    }

    // Prepare data for the chart
    const data = allEvents.map(event => ({
        x: event.time,
        y: event.count
    }));

    createOrUpdateTimelineChart(ctx, data);
}

// Get request events from strategies
function getRequestEvents() {
    for (const strategy in incrementalResults.strategies) {
        const events = incrementalResults.strategies[strategy].events || [];
        const requestEvents = events.filter(e => e.type === 'new_requests');

        if (requestEvents.length > 0) {
            return requestEvents;
        }
    }

    return [];
}

// Create empty timeline chart when no data
function createEmptyTimelineChart(ctx) {
    if (charts.requestTimelineChart) {
        charts.requestTimelineChart.destroy();
    }

    charts.requestTimelineChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Requests',
                data: [],
                backgroundColor: 'rgba(75, 192, 192, 0.6)'
            }]
        },
        options: getTimelineChartOptions()
    });
}

// Create or update request timeline chart
function createOrUpdateTimelineChart(ctx, data) {
    if (charts.requestTimelineChart) {
        charts.requestTimelineChart.data.datasets[0].data = data;
        charts.requestTimelineChart.update();
    } else {
        charts.requestTimelineChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'New Requests',
                    data,
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }]
            },
            options: getTimelineChartOptions()
        });
    }
}

// Get options for timeline chart
function getTimelineChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0
        },
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: {
                    display: true,
                    text: 'Time (seconds)'
                }
            },
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Request Count'
                }
            }
        },
        plugins: {
            tooltip: {
                callbacks: {
                    label: function(context) {
                        return `Time: ${context.parsed.x.toFixed(1)}s, Requests: ${context.parsed.y}`;
                    }
                }
            }
        }
    };
}

// Update system load chart
function updateSystemLoadChart() {
    const ctx = document.getElementById('system-load-chart')?.getContext('2d');
    if (!ctx) return;

    const { timeMetrics, simulationTime } = getSystemLoadData();

    // If no metrics, show empty chart
    if (!timeMetrics || timeMetrics.length === 0) {
        createEmptySystemLoadChart(ctx);
        return;
    }

    createOrUpdateSystemLoadChart(ctx, timeMetrics);
}

// Get system load data from strategies
function getSystemLoadData() {
    for (const strategy in incrementalResults.strategies) {
        const metrics = incrementalResults.strategies[strategy].time_metrics || [];
        const simTime = incrementalResults.strategies[strategy].simulation_time || 0;

        if (metrics.length > 0) {
            return {
                timeMetrics: metrics,
                simulationTime: simTime
            };
        }
    }

    return {
        timeMetrics: [],
        simulationTime: 0
    };
}

// Create empty system load chart
function createEmptySystemLoadChart(ctx) {
    if (charts.systemLoadChart) {
        charts.systemLoadChart.destroy();
    }

    charts.systemLoadChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

// Create or update system load chart
function createOrUpdateSystemLoadChart(ctx, timeMetrics) {
    // Prepare data
    const times = timeMetrics.map(metric => metric.time);
    const activeRequests = timeMetrics.map(metric => metric.active_requests || 0);
    const availableTransporters = timeMetrics.map(metric => metric.available_transporters || 0);

    if (charts.systemLoadChart) {
        charts.systemLoadChart.data.labels = times;
        charts.systemLoadChart.data.datasets[0].data = activeRequests;
        charts.systemLoadChart.data.datasets[1].data = availableTransporters;
        charts.systemLoadChart.update();
    } else {
        charts.systemLoadChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: times,
                datasets: [
                    {
                        label: 'Active Requests',
                        data: activeRequests,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Available Transporters',
                        data: availableTransporters,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.1,
                        yAxisID: 'y'
                    }
                ]
            },
            options: getSystemLoadChartOptions()
        });
    }
}

// Get options for system load chart
function getSystemLoadChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Time (seconds)'
                }
            },
            y: {
                beginAtZero: true,
                position: 'left',
                title: {
                    display: true,
                    text: 'Count'
                }
            }
        },
        plugins: {
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const label = context.dataset.label || '';
                        const value = context.parsed.y;
                        return `${label}: ${value}`;
                    }
                }
            }
        }
    };
}

// Update strategy comparison chart
function updateStrategyComparisonChart() {
    const ctx = document.getElementById('strategy-comparison-chart')?.getContext('2d');
    if (!ctx) return;

    // Collect makespan estimates from all strategies
    const datasets = createStrategyComparisonDatasets();

    // If no datasets, show empty chart
    if (datasets.length === 0) {
        createEmptyStrategyComparisonChart(ctx);
        return;
    }

    createOrUpdateStrategyComparisonChart(ctx, datasets);
}

// Create datasets for strategy comparison
function createStrategyComparisonDatasets() {
    const datasets = [];
    const strategyColors = getStrategyColors();

    for (const strategy in incrementalResults.strategies) {
        const metrics = incrementalResults.strategies[strategy].time_metrics || [];

        if (metrics.length === 0) continue;

        // Get times and makespan estimates
        const times = metrics.map(metric => metric.time);
        const makespans = metrics.map(metric => metric.makespan_estimate || 0);

        // Create dataset
        datasets.push({
            label: strategy,
            data: makespans.map((val, i) => ({ x: times[i], y: val })),
            borderColor: strategyColors[strategy] || `hsl(${Math.random() * 360}, 70%, 50%)`,
            backgroundColor: `${strategyColors[strategy] || `hsl(${Math.random() * 360}, 70%, 50%)`}33`,
            tension: 0.1
        });
    }

    return datasets;
}

// Define colors for each strategy
function getStrategyColors() {
    return {
        'ILP: Makespan': 'rgba(75, 108, 183, 1)',
        'ILP: Equal Workload': 'rgba(241, 196, 15, 1)',
        'ILP: Urgency First': 'rgba(230, 126, 34, 1)',
        'ILP: Cluster-Based': 'rgba(46, 204, 113, 1)',
        'Genetic Algorithm': 'rgba(155, 89, 182, 1)',
        'Random': 'rgba(231, 76, 60, 1)'
    };
}

// Create empty strategy comparison chart
function createEmptyStrategyComparisonChart(ctx) {
    if (charts.strategyComparisonChart) {
        charts.strategyComparisonChart.destroy();
    }

    charts.strategyComparisonChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

// Create or update strategy comparison chart
function createOrUpdateStrategyComparisonChart(ctx, datasets) {
    if (charts.strategyComparisonChart) {
        charts.strategyComparisonChart.data.datasets = datasets;
        charts.strategyComparisonChart.update();
    } else {
        charts.strategyComparisonChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets
            },
            options: getStrategyComparisonChartOptions()
        });
    }
}

// Get options for strategy comparison chart
function getStrategyComparisonChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0
        },
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: {
                    display: true,
                    text: 'Time (seconds)'
                }
            },
            y: {
                type: 'linear',
                position: 'left',
                title: {
                    display: true,
                    text: 'Makespan Estimate (seconds)'
                }
            }
        },
        plugins: {
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const label = context.dataset.label || '';
                        const value = context.parsed.y;
                        return `${label}: ${value.toFixed(1)}s`;
                    }
                }
            }
        }
    };
}

// Function to zoom timeline charts
function zoomTimelineCharts(mode) {
    if (mode === 'all') {
        resetChartZoom();
    } else if (mode === 'peak') {
        zoomToPeakPeriod();
    }
}

// Reset zoom on all charts
function resetChartZoom() {
    const charts = [
        'requestTimelineChart',
        'systemLoadChart',
        'strategyComparisonChart'
    ];

    charts.forEach(chartName => {
        if (window.charts && window.charts[chartName]) {
            window.charts[chartName].options.scales.x.min = undefined;
            window.charts[chartName].options.scales.x.max = undefined;
            window.charts[chartName].update();
        }
    });
}

// Zoom to peak activity period
function zoomToPeakPeriod() {
    // Find peak period
    const { peakStart, peakEnd } = findPeakPeriod();

    // Apply zoom to all charts
    const charts = [
        'requestTimelineChart',
        'systemLoadChart',
        'strategyComparisonChart'
    ];

    charts.forEach(chartName => {
        if (window.charts && window.charts[chartName]) {
            window.charts[chartName].options.scales.x.min = peakStart;
            window.charts[chartName].options.scales.x.max = peakEnd;
            window.charts[chartName].update();
        }
    });
}

// Find period with peak activity
function findPeakPeriod() {
    let peakStart = 0;
    let peakEnd = 0;

    // Use time metrics from one of the strategies
    for (const strategy in incrementalResults.strategies) {
        const metrics = incrementalResults.strategies[strategy].time_metrics || [];

        if (metrics.length > 0) {
            // Find the time with the highest number of active requests
            let maxActiveRequests = 0;
            let peakTime = 0;

            metrics.forEach(metric => {
                if (metric.active_requests > maxActiveRequests) {
                    maxActiveRequests = metric.active_requests;
                    peakTime = metric.time;
                }
            });

            // Set window around peak (±2 minutes)
            peakStart = Math.max(0, peakTime - 120);
            peakEnd = peakTime + 120;

            break;
        }
    }

    return { peakStart, peakEnd };
}

// Get incremental mode configuration
function getIncrementalModeConfig() {
    const isIncrementalMode = incrementalToggle?.checked || false;

    if (!isIncrementalMode) {
        return { incremental_mode: false };
    }

    // Get time range in seconds (convert from minutes)
    const timeRangeMinutes = parseInt(document.getElementById('timeRangeSlider')?.value || '60');
    const timeRangeSeconds = timeRangeMinutes * 60;

    // Get time distribution
    const timeDistribution = document.querySelector('input[name="timeDistribution"]:checked')?.value || 'realistic';

    return {
        incremental_mode: true,
        time_range: [0, timeRangeSeconds],
        time_distribution: timeDistribution
    };
}

// Load hourly rate data from server
function loadHourlyRateData() {
    fetch('/get_hourly_rate_data')
        .then(response => response.json())
        .then(data => {
            createHourlyRateChart(data.hourly_rates);
        })
        .catch(error => {
            console.error('Error loading hourly rate data:', error);
        });
}

// Create hourly rate chart
function createHourlyRateChart(hourlyRates) {
    const ctx = document.getElementById('hourly-rate-chart')?.getContext('2d');
    if (!ctx) return;

    if (!hourlyRates || !hourlyRates.labels || !hourlyRates.data) {
        console.error('Invalid hourly rate data');
        return;
    }

    if (window.hourlyRateChart) {
        window.hourlyRateChart.destroy();
    }

    window.hourlyRateChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hourlyRates.labels,
            datasets: [{
                label: 'Requests per Hour',
                data: hourlyRates.data,
                backgroundColor: 'rgba(75, 108, 183, 0.7)',
                borderColor: 'rgba(75, 108, 183, 1)',
                borderWidth: 1
            }]
        },
        options: getHourlyRateChartOptions()
    });
}

// Get options for hourly rate chart
function getHourlyRateChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: {
                display: false
            },
            legend: {
                display: false
            },
            tooltip: {
                callbacks: {
                    title: function(tooltipItems) {
                        return tooltipItems[0].label;
                    },
                    label: function(context) {
                        return `${context.raw.toFixed(2)} requests/hour`;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: false
                }
            },
            x: {
                title: {
                    display: false
                }
            }
        }
    };
}

// Override startBenchmark to include incremental mode
const originalStartBenchmark = startBenchmark;
startBenchmark = function() {
    // Reset incremental results
    incrementalResults = {
        strategies: {},
        hasData: false
    };

    // Get configuration values
    const numTransporters = parseInt(transporterCountSlider.value);
    const randomRuns = parseInt(randomRunsSlider.value);
    const strategies = getSelectedStrategies();
    const scenarios = getSelectedScenarios();

    // Get incremental mode configuration
    const incrementalConfig = getIncrementalModeConfig();

    // Validate configuration
    if (strategies.length === 0) {
        notifyUser('Please select at least one strategy.', 'error');
        return;
    }

    if (scenarios.length === 0) {
        notifyUser('Please select at least one scenario.', 'error');
        return;
    }

    // Clear previous results
    clearBenchmarkResults();

    // Show progress modal
    const progressModal = document.getElementById('benchmark-progress-modal');
    progressModal.style.display = 'block';

    // Update status
    document.getElementById('benchmark-status').textContent = incrementalConfig.incremental_mode ?
        'Status: Running Incremental Benchmark' : 'Status: Running Benchmark';
    benchmarkRunning = true;
    runBenchmarkBtn.disabled = true;

    // Initialize progress bar
    updateProgressBar(0, 'Initializing benchmark...');

    // Send benchmark request to server
    sendBenchmarkRequest(numTransporters, randomRuns, strategies, scenarios, incrementalConfig);
};

// Send benchmark request to server
function sendBenchmarkRequest(numTransporters, randomRuns, strategies, scenarios, incrementalConfig) {
    fetch('/start_benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            transporters: numTransporters,
            random_runs: randomRuns,
            strategies: strategies,
            scenarios: scenarios,
            ...incrementalConfig
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        notifyUser('Benchmark started successfully.', 'success');
    })
    .catch(error => {
        notifyUser(`Failed to start benchmark: ${error}`, 'error');
        finalizeBenchmark({ error: error.message });
    });
}