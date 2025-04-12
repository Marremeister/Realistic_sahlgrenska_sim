"""
Benchmark package for hospital transport system.
Provides tools to compare different transport assignment strategies.
"""


def setup_benchmark(app, socketio, hospital_system):
    """
    Initialize and connect all benchmark components.

    Args:
        app: Flask application instance
        socketio: Flask-SocketIO instance
        hospital_system: Hospital system instance

    Returns:
        BenchmarkController: The initialized benchmark controller
    """
    # Create directories (if they don't exist)
    import os
    for dir_path in ["model", "view", "controller", "repository"]:
        os.makedirs(os.path.join(os.path.dirname(__file__), dir_path), exist_ok=True)

    # Import components
    from new_backend_benchmark.model.benchmark_model import BenchmarkModel
    from new_backend_benchmark.controller.benchmark_controller import BenchmarkController
    from new_backend_benchmark.view.benchmark_view import BenchmarkView

    # Create MVC components
    model = BenchmarkModel(hospital_system)
    controller = BenchmarkController(model, socketio)
    view = BenchmarkView(app, socketio, controller)

    # Register routes
    view.register_routes()

    print("✅ Benchmark MVC initialized")
    return controller