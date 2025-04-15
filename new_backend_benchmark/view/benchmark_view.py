"""
View component for benchmark functionality.
Handles presentation and user interaction for benchmark operations.
"""

from flask import request, jsonify, render_template
from typing import Dict, Any, List, Tuple, Optional


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
        self._register_page_routes()
        self._register_api_routes()
        self._register_time_based_routes()
        self._register_socketio_handlers()
        print("✅ Benchmark routes registered")

    def _register_page_routes(self):
        """Register page routes."""
        self.app.add_url_rule("/benchmark", "benchmark_page", self._serve_benchmark_page)

    def _register_api_routes(self):
        """Register API routes."""
        self.app.add_url_rule("/start_benchmark", "start_benchmark",
                              self._start_benchmark, methods=["POST"])
        self.app.add_url_rule("/cancel_benchmark", "cancel_benchmark",
                              self._cancel_benchmark, methods=["POST"])
        self.app.add_url_rule("/get_scenarios", "get_scenarios",
                              self._get_scenarios)
        self.app.add_url_rule("/get_strategies", "get_strategies",
                              self._get_strategies)
        self.app.add_url_rule("/add_scenario", "add_scenario",
                              self._add_scenario, methods=["POST"])

    def _register_time_based_routes(self):
        """Register time-based benchmark routes."""
        self.app.add_url_rule("/get_available_time_ranges", "get_available_time_ranges",
                              self._get_available_time_ranges)
        self.app.add_url_rule("/run_time_based_benchmark", "run_time_based_benchmark",
                              self._run_time_based_benchmark, methods=["POST"])
        self.app.add_url_rule("/generate_time_scenario", "generate_time_scenario",
                              self._generate_time_scenario, methods=["POST"])
        self.app.add_url_rule("/get_hourly_rate_data", "get_hourly_rate_data",
                              self._get_hourly_rate_data, methods=["GET"])

    # Route handler methods

    def _serve_benchmark_page(self):
        """Serve the benchmark page."""
        try:
            return render_template('benchmark.html')
        except Exception as e:
            print(f"Error serving benchmark.html: {e}")
            return f"Error serving benchmark page: {e}", 404

    def _start_benchmark(self):
        """API route to start a benchmark."""
        data = self._get_json_data()
        if not data:
            return self._error_response("No configuration provided"), 400

        result = self.controller.start_benchmark(data)
        return jsonify(result)

    def _cancel_benchmark(self):
        """API route to cancel a running benchmark."""
        result = self.controller.cancel_benchmark()
        return jsonify(result)

    def _get_scenarios(self):
        """API route to get available benchmark scenarios."""
        scenarios = self.controller.get_available_scenarios()
        return jsonify(scenarios)

    def _get_strategies(self):
        """API route to get available benchmark strategies."""
        strategies = self.controller.get_available_strategies()
        return jsonify(strategies)

    def _add_scenario(self):
        """API route to add a custom benchmark scenario."""
        data = self._get_json_data()
        if not self._validate_scenario_data(data):
            return self._error_response("Invalid scenario data"), 400

        name = data['name']
        requests = data['requests']
        scenarios = self.controller.add_custom_scenario(name, requests)

        return jsonify(self._success_response(
            {"scenarios": scenarios},
            "Scenario added"
        ))

    def _get_available_time_ranges(self):
        """API route to get available time ranges and hourly rates."""
        result = self.controller.get_available_time_ranges()
        return jsonify(result)

    def _generate_time_scenario(self):
        """API route to generate a time-based scenario."""
        data = self._get_json_data()
        if not data:
            return self._error_response("No data provided"), 400

        # Extract parameters
        params = self._extract_time_scenario_params(data)

        # Generate scenario
        result = self.controller.generate_time_scenario(**params)

        if not result.get("success", False):
            return self._error_response(result.get("error", "Unknown error")), 400

        return jsonify(result)

    def _run_time_based_benchmark(self):
        """API route to run a time-based benchmark."""
        data = self._get_json_data()
        if not data:
            return self._error_response("No data provided"), 400

        # Extract parameters
        params = self._extract_time_benchmark_params(data)

        # Run benchmark
        result = self.controller.run_time_based_benchmark(**params)

        if not result.get("success", False):
            return self._error_response(result.get("error", "Unknown error")), 400

        return jsonify(result["data"])

    def _get_hourly_rate_data(self):
        """
        Get hourly rate data for time-based benchmarks.

        Returns:
            JSON: Hourly rate data for visualization
        """
        data = self.controller.get_hourly_rate_data()
        return jsonify(data)

    # SocketIO event handlers

    def _register_socketio_handlers(self):
        """Register SocketIO event handlers for real-time communication."""

        @self.socketio.on('connect')
        def handle_connect():
            print("Client connected to benchmark socket")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print("Client disconnected from benchmark socket")

        @self.socketio.on('benchmark_request')
        def handle_benchmark_request(data):
            """Handle a benchmark request from the client."""
            print(f"Received benchmark request: {data}")
            result = self.controller.start_benchmark(data)
            return result

        @self.socketio.on('incremental_benchmark_results')
        def handle_incremental_results(data):
            """Handle and broadcast incremental benchmark results."""
            # This just re-emits the data to all clients
            self.socketio.emit('incremental_benchmark_results', data)

    # Helper methods

    def _get_json_data(self) -> Optional[Dict[str, Any]]:
        """
        Get JSON data from request.

        Returns:
            Optional[Dict[str, Any]]: JSON data or None if invalid
        """
        try:
            return request.get_json()
        except Exception:
            return None

    def _validate_scenario_data(self, data: Optional[Dict[str, Any]]) -> bool:
        """
        Validate scenario data.

        Args:
            data: Scenario data to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not data or 'name' not in data or 'requests' not in data:
            return False

        if not isinstance(data['requests'], list):
            return False

        return True

    def _extract_time_scenario_params(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameters for time scenario generation.

        Args:
            data: Request data

        Returns:
            Dict[str, Any]: Parameter dictionary
        """
        return {
            "start_hour": data.get("start_hour"),
            "end_hour": data.get("end_hour"),
            "name": data.get("name"),
            "request_count": data.get("request_count")
        }

    def _extract_time_benchmark_params(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameters for time-based benchmark.

        Args:
            data: Request data

        Returns:
            Dict[str, Any]: Parameter dictionary
        """
        return {
            "start_hour": data.get("start_hour"),
            "end_hour": data.get("end_hour"),
            "transporters": data.get("transporter_count", 3),
            "random_runs": data.get("random_runs", 100)
        }

    def _error_response(self, message: str) -> Dict[str, str]:
        """
        Create a standardized error response.

        Args:
            message: Error message

        Returns:
            Dict: Error response dictionary
        """
        return {
            "status": "error",
            "message": message
        }

    def _success_response(self, data: Any = None, message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a standardized success response.

        Args:
            data: Response data (optional)
            message: Success message (optional)

        Returns:
            Dict: Success response dictionary
        """
        response = {"status": "success"}

        if message:
            response["message"] = message

        if data is not None:
            response["data"] = data

        return response