from datetime import datetime, timezone
import json

try:
    from redis import Redis
except ModuleNotFoundError:
    Redis = None

try:
    from rq import Queue, Worker
except Exception:
    Queue = None
    Worker = None


def init_background_queue(app):
    app.extensions["analysis_queue"] = None
    app.extensions["redis_conn"] = None

    redis_url = app.config.get("REDIS_URL")
    if not redis_url or Redis is None or Queue is None:
        return

    try:
        redis_conn = Redis.from_url(redis_url)
        redis_conn.ping()
        queue = Queue("analysis", connection=redis_conn)
        app.extensions["redis_conn"] = redis_conn
        app.extensions["analysis_queue"] = queue
    except Exception:
        app.extensions["redis_conn"] = None
        app.extensions["analysis_queue"] = None


def get_analysis_queue(app):
    return app.extensions.get("analysis_queue")


def enqueue_analysis_job(app, job_id: int):
    queue = get_analysis_queue(app)
    if queue is None:
        return None
    return queue.enqueue("app.background.run_analysis_job", job_id, job_timeout="30m")


def run_analysis_job(job_id: int):
    from app import create_app, mysql, nlp_utils
    from app.notifications import send_email_alert, send_slack_alert
    from psycopg2.extras import RealDictCursor

    app = create_app()
    with app.app_context():
        cursor = mysql.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT job_id, job_type, payload, status, requested_by
            FROM analysis_jobs
            WHERE job_id = %s
            FOR UPDATE
        """, (job_id,))
        job = cursor.fetchone()
        if not job or job["status"] not in {"queued", "running"}:
            cursor.close()
            return

        cursor.execute("""
            UPDATE analysis_jobs
            SET status = 'running', started_at = %s, error_message = NULL
            WHERE job_id = %s
        """, (datetime.now(timezone.utc), job_id))
        mysql.connection.commit()

        try:
            if job["job_type"] == "reanalysis":
                payload = job.get("payload") or {}
                scope = payload.get("scope", "all") if isinstance(payload, dict) else "all"
                if scope == "recent":
                    cursor.execute("""
                        SELECT review_id, review_text, overall_sentiment, overall_sentiment_score
                        FROM reviews
                        ORDER BY uploaded_at DESC
                        LIMIT 200
                    """)
                else:
                    cursor.execute("""
                        SELECT review_id, review_text, overall_sentiment, overall_sentiment_score
                        FROM reviews
                        ORDER BY uploaded_at DESC
                    """)
                reviews = cursor.fetchall()
                for review in reviews:
                    nlp_utils.analyze_review_detailed(
                        review["review_text"],
                        review["overall_sentiment"],
                        review["overall_sentiment_score"] or 0.0,
                        mysql=mysql,
                        review_id=review["review_id"]
                    )
                child_job_id = _create_alert_scan_job(cursor, job.get("requested_by") or "system")
                mysql.connection.commit()
                queue = app.extensions.get("analysis_queue")
                if queue is not None:
                    queue.enqueue("app.background.run_analysis_job", child_job_id, job_timeout="15m")
            elif job["job_type"] == "alert_scan":
                payload = job.get("payload") or {}
                scope = payload.get("scope", "recent") if isinstance(payload, dict) else "recent"
                if scope == "all":
                    cursor.execute("""
                        SELECT r.review_id, r.review_text, r.overall_sentiment, r.overall_sentiment_score, ras.analysis_payload
                        FROM reviews r
                        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
                        ORDER BY r.uploaded_at DESC
                        LIMIT 300
                    """)
                else:
                    cursor.execute("""
                        SELECT r.review_id, r.review_text, r.overall_sentiment, r.overall_sentiment_score, ras.analysis_payload
                        FROM reviews r
                        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
                        ORDER BY r.uploaded_at DESC
                        LIMIT 120
                    """)
                review_rows = cursor.fetchall()
                analyses = []
                for row in review_rows:
                    payload_data = row.get("analysis_payload")
                    if payload_data:
                        analysis = json.loads(payload_data) if isinstance(payload_data, str) else payload_data
                    else:
                        analysis = nlp_utils.analyze_review_detailed(
                            row["review_text"],
                            row["overall_sentiment"],
                            row["overall_sentiment_score"] or 0.0,
                            mysql=mysql,
                            review_id=row["review_id"]
                        )
                    analysis["review_id"] = row["review_id"]
                    analyses.append(analysis)

                for alert in nlp_utils.generate_alert_candidates(analyses):
                    cursor.execute("""
                        SELECT alert_id
                        FROM analysis_alerts
                        WHERE alert_type = %s AND title = %s AND status = 'open'
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (alert["alert_type"], alert["title"]))
                    existing = cursor.fetchone()
                    if existing:
                        continue

                    delivered_slack = False
                    delivered_email = False
                    try:
                        delivered_slack = send_slack_alert(
                            app.config.get("SLACK_WEBHOOK_URL"),
                            alert["title"],
                            alert["message"]
                        )
                    except Exception:
                        delivered_slack = False
                    try:
                        delivered_email = send_email_alert(app.config, alert["title"], alert["message"])
                    except Exception:
                        delivered_email = False

                    cursor.execute("""
                        INSERT INTO analysis_alerts (
                            alert_type, severity, title, message, payload,
                            delivered_slack, delivered_email
                        )
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                    """, (
                        alert["alert_type"],
                        alert["severity"],
                        alert["title"],
                        alert["message"],
                        json.dumps(alert.get("payload") or {}),
                        delivered_slack,
                        delivered_email
                    ))
            else:
                raise ValueError(f"Unsupported job type: {job['job_type']}")

            cursor.execute("""
                UPDATE analysis_jobs
                SET status = 'completed', completed_at = %s
                WHERE job_id = %s
            """, (datetime.now(timezone.utc), job_id))
            mysql.connection.commit()
        except Exception as exc:
            mysql.connection.rollback()
            cursor.execute("""
                UPDATE analysis_jobs
                SET status = 'failed', completed_at = %s, error_message = %s
                WHERE job_id = %s
            """, (datetime.now(timezone.utc), str(exc), job_id))
            mysql.connection.commit()
            raise
        finally:
            cursor.close()


def get_worker(app):
    redis_conn = app.extensions.get("redis_conn")
    if redis_conn is None or Worker is None:
        raise RuntimeError("RQ worker is not configured")
    return Worker(["analysis"], connection=redis_conn)


def _create_alert_scan_job(cursor, requested_by: str):
    cursor.execute("""
        INSERT INTO analysis_jobs (job_type, requested_by, payload)
        VALUES ('alert_scan', %s, %s::jsonb)
        RETURNING job_id
    """, (requested_by, json.dumps({"scope": "recent"})))
    row = cursor.fetchone()
    return row["job_id"]
