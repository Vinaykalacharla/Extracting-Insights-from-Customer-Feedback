import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402
from app.background import get_worker  # noqa: E402


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        worker = get_worker(app)
        worker.work()
