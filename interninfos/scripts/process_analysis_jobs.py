import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, mysql  # noqa: E402
from app.background import run_analysis_job  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402


def process_once():
    app = create_app()
    with app.app_context():
        cursor = mysql.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT job_id
            FROM analysis_jobs
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
        """)
        row = cursor.fetchone()
        cursor.close()
        if not row:
            return False
        run_analysis_job(row["job_id"])
        return True


if __name__ == "__main__":
    processed = process_once()
    print("processed" if processed else "no_jobs")
