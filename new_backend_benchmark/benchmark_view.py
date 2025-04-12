"""
View component for benchmark functionality.
Handles presentation and user interaction for benchmark operations.
"""


from flask import request, jsonify, render_template



class BenchmarkView:
    """
    View class for benchmark functionality.
    Handles routes and presenting benchmark results to users.
    """

    def __init__(self, app, socketio, controller):
        """
        Initialize the benchmark view.

        Args:
            app: Flask application instance
            socketio: Flask-SocketIO instance
            controller: BenchmarkController instance
        """
        self.app = app
        self.socketio = socketio
        self.controller = controller

    def register_routes(self):
        """Register all routes for the benchmark functionality."""
        # Page route
        self.app.add_url_rule("/benchmark", "benchmark_page", self.benchmark_page)

        # API routes
        self.app.add_url_rule("/start_benchmark", "start_benchmark",
                              self.start_benchmark, methods=["POST"])
        self.app.add_url_rule("/cancel_benchmark", "cancel_benchmark",
                              self.cancel_benchmark, methods=["POST"])
        self.app.add_url_rule("/get_scenarios", "get_scenarios",
                              self.get_scenarios)
        self.app.add_url_rule("/add_scenario", "add_scenario",
                              self.add_scenario, methods=["POST"])

        # Time-based benchmark routes
        self.app.add_url_rule("/get_available_time_ranges", "get_available_time_ranges",
                              self.get_available_time_ranges)
        self.app.add_url_rule("/run_time_based_benchmark", "run_time_based_benchmark",
                              self.run_time_based_benchmark, methods=["POST"])
        self.app.add_url_rule("/generate_time_scenario", "generate_time_scenario",
                              self.generate_time_scenario, methods=["POST"])

        # Register SocketIO event handlers
        self._register_socketio_handlers()

        print("✅ Benchmark routes registered")

    # In benchmark_view.py
    def benchmark_page(self):
        """Serve the benchmark page."""
        try:

            return render_template('benchmark.html')
        except Exception as e:
            print(f"Error serving benchmark.html: {e}")
            return f"Error serving benchmark page: {e}", 404

    def start_benchmark(self):
        """API route to start a benchmark."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration provided"}), 400

        result = self.controller.start_benchmark(data)
        return jsonify(result)

    def cancel_benchmark(self):
        """API route to cancel a running benchmark."""
        result = self.controller.cancel_benchmark()
        return jsonify(result)

    def get_scenarios(self):
        """API route to get available benchmark scenarios."""
        scenarios = self.controller.get_available_scenarios()
        return jsonify(scenarios)

    def add_scenario(self):
        """API route to add a custom benchmark scenario."""
        data = request.get_json()
        if not data or 'name' not in data or 'requests' not in data:
            return jsonify({"error": "Invalid scenario data"}), 400

        name = data['name']
        requests = data['requests']

        if not isinstance(requests, list):
            return jsonify({"error": "Requests must be a list"}), 400

        scenarios = self.controller.add_custom_scenario(name, requests)
        return jsonify({"status": "Scenario added", "scenarios": scenarios})

    def _register_socketio_handlers(self):
        """Register SocketIO event handlers for real-time communication."""

        @self.socketio.on('benchmark_request')
        def handle_benchmark_request(data):
            """Handle a benchmark request from the client."""
            print(f"Received benchmark request: {data}")
            result = self.controller.start_benchmark(data)
            return result

    def _register_socketio_handlers(self):
        """Register SocketIO event handlers for real-time communication."""

        @self.socketio.on('benchmark_request')
        def handle_benchmark_request(data):
            """Handle a benchmark request from the client."""
            print(f"Received benchmark request: {data}")
            result = self.controller.start_benchmark(data)
            return result

    # Time-based benchmark route handlers

    def get_available_time_ranges(self):
        """API route to get available time ranges and hourly rates."""
        time_ranges = self.controller.get_available_time_ranges()
        hourly_rates = self.controller.get_hourly_rate_data()

        return jsonify({
            "time_ranges": time_ranges,
            "hourly_rates": hourly_rates
        })

    def generate_time_scenario(self):
        """API route to generate a time-based scenario."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract parameters
        start_hour = data.get("start_hour")
        end_hour = data.get("end_hour")
        name = data.get("name")
        request_count = data.get("request_count")

        # Generate scenario
        result = self.controller.generate_time_scenario(
            start_hour, end_hour, name, request_count
        )

        if not result.get("success", False):
            return jsonify({"error": result.get("error", "Unknown error")}), 400

        return jsonify(result)

    def run_time_based_benchmark(self):
        """API route to run a time-based benchmark."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract parameters
        start_hour = data.get("start_hour")
        end_hour = data.get("end_hour")
        transporter_count = data.get("transporter_count", 3)
        random_runs = data.get("random_runs", 100)

        # Run benchmark
        result = self.controller.run_time_based_benchmark(
            start_hour, end_hour, transporter_count, random_runs
        )

        if not result.get("success", False):
            return jsonify({"error": result.get("error", "Unknown error")}), 400

        return jsonify(result["data"])