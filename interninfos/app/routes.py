from flask import Blueprint, render_template, request, redirect, session, url_for, flash, current_app, send_file, jsonify
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
from collections import Counter
import csv
import io
import json
import os
from psycopg2.extras import RealDictCursor

from . import mysql  # initialized in init.py
from . import nlp_utils  # Import NLP utilities
from .background import enqueue_analysis_job as enqueue_rq_analysis_job, get_analysis_queue

# Import specific functions from nlp_utils to avoid conflicts
from .nlp_utils import map_sentiment, highlight_keywords, preprocess_text

# Import for PDF generation
from reportlab.graphics.charts.barcharts import VerticalBarChart

main = Blueprint('main', __name__, url_prefix="/")

# ---------- Helpers ----------
def dict_cursor():
    return mysql.connection.cursor(cursor_factory=RealDictCursor)


def fetch_review_analysis_payloads(where_sql="", params=(), limit=100):
    cursor = dict_cursor()
    query = f"""
        SELECT r.review_id, r.review_text, r.overall_sentiment, r.overall_sentiment_score, r.uploaded_at,
               ras.analysis_payload
        FROM reviews r
        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
        {where_sql}
        ORDER BY r.uploaded_at DESC
        LIMIT %s
    """
    cursor.execute(query, (*params, limit))
    rows = cursor.fetchall()
    cursor.close()

    analyses = []
    for row in rows:
        payload = row.get('analysis_payload')
        if payload:
            analysis = json.loads(payload) if isinstance(payload, str) else payload
        else:
            analysis = nlp_utils.analyze_review_detailed(
                row['review_text'],
                row['overall_sentiment'],
                row['overall_sentiment_score'] or 0.0,
                mysql=mysql,
                review_id=row['review_id']
            )
        analysis['review_id'] = row['review_id']
        analysis['uploaded_at'] = row.get('uploaded_at')
        analysis['review_text'] = row['review_text']
        analysis['overall_sentiment'] = row['overall_sentiment']
        analyses.append(analysis)

    return analyses


def enqueue_analysis_job(job_type: str, requested_by: str, payload: dict | None = None):
    cursor = dict_cursor()
    cursor.execute("""
        INSERT INTO analysis_jobs (job_type, requested_by, payload)
        VALUES (%s, %s, %s::jsonb)
        RETURNING job_id, status, created_at
    """, (job_type, requested_by, json.dumps(payload or {})))
    job = cursor.fetchone()
    mysql.connection.commit()
    cursor.close()
    return job


VALID_SENTIMENT_FILTERS = {"all", "positive", "negative", "neutral"}
VALID_URGENCY_FILTERS = {"all", "low", "medium", "high"}
TREND_RANGE_OPTIONS = {
    "14": {"days": 14, "label": "Last 14 days"},
    "30": {"days": 30, "label": "Last 30 days"},
    "90": {"days": 90, "label": "Last 90 days"},
    "180": {"days": 180, "label": "Last 180 days"},
    "all": {"days": None, "label": "All time"},
}


def normalize_sentiment(sentiment: str | None) -> str:
    normalized = (sentiment or "neutral").strip().lower()
    return normalized if normalized in {"positive", "negative", "neutral"} else "neutral"


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def password_min_length() -> int:
    return int(current_app.config.get("MIN_PASSWORD_LENGTH", 8))


def password_rule_error(new_password: str, confirm_password: str) -> str | None:
    if new_password != confirm_password:
        return "Passwords do not match."
    if len(new_password) < password_min_length():
        return f"Password must be at least {password_min_length()} characters long."
    return None


def parse_review_filters(args):
    search_term = (args.get("q") or "").strip()
    sentiment = (args.get("sentiment") or "all").strip().lower()
    urgency = (args.get("urgency") or "all").strip().lower()
    intent = (args.get("intent") or "all").strip().lower()

    if sentiment not in VALID_SENTIMENT_FILTERS:
        sentiment = "all"
    if urgency not in VALID_URGENCY_FILTERS:
        urgency = "all"
    if not intent:
        intent = "all"

    return {
        "q": search_term,
        "sentiment": sentiment,
        "urgency": urgency,
        "intent": intent,
    }


def build_user_review_where_sql(user_id, filters):
    where_clauses = ["r.user_id = %s"]
    params = [user_id]

    if filters["q"]:
        where_clauses.append("r.review_text ILIKE %s")
        params.append(f"%{filters['q']}%")
    if filters["sentiment"] != "all":
        where_clauses.append("LOWER(COALESCE(r.overall_sentiment, 'neutral')) = %s")
        params.append(filters["sentiment"])
    if filters["urgency"] != "all":
        where_clauses.append("LOWER(COALESCE(ras.urgency_level, 'low')) = %s")
        params.append(filters["urgency"])
    if filters["intent"] != "all":
        where_clauses.append("LOWER(COALESCE(ras.intent_label, 'unknown')) = %s")
        params.append(filters["intent"])

    return f"WHERE {' AND '.join(where_clauses)}", tuple(params)


def parse_trend_range(raw_value: str | None):
    range_key = (raw_value or "30").strip().lower()
    if range_key not in TREND_RANGE_OPTIONS:
        range_key = "30"
    return range_key, TREND_RANGE_OPTIONS[range_key]


def fetch_review_rollup(cursor, base_where_clauses, base_params, uploaded_after=None, uploaded_before=None):
    where_clauses = list(base_where_clauses)
    params = list(base_params)

    if uploaded_after is not None:
        where_clauses.append("r.uploaded_at >= %s")
        params.append(uploaded_after)
    if uploaded_before is not None:
        where_clauses.append("r.uploaded_at < %s")
        params.append(uploaded_before)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    cursor.execute(f"""
        SELECT
            COUNT(*) AS total_reviews,
            SUM(CASE WHEN LOWER(COALESCE(r.overall_sentiment, 'neutral')) = 'positive' THEN 1 ELSE 0 END) AS positive_reviews,
            SUM(CASE WHEN LOWER(COALESCE(r.overall_sentiment, 'neutral')) = 'negative' THEN 1 ELSE 0 END) AS negative_reviews,
            SUM(CASE WHEN LOWER(COALESCE(r.overall_sentiment, 'neutral')) = 'neutral' THEN 1 ELSE 0 END) AS neutral_reviews,
            AVG(r.overall_sentiment_score) AS avg_sentiment_score
        FROM reviews r
        {where_sql}
    """, tuple(params))
    return cursor.fetchone() or {}


def build_priority_queue(analyses, limit=4):
    queue = []

    for analysis in analyses:
        advanced = analysis.get("advanced_insights", {}) or {}
        urgency = advanced.get("urgency", {}) or {}
        sentiment = normalize_sentiment(analysis.get("overall_sentiment"))
        urgency_level = (urgency.get("level") or "low").lower()
        impact_score = round(safe_float(advanced.get("impact_score")), 1)
        experience_score = round(safe_float(advanced.get("experience_score")), 1)
        needs_action = urgency_level == "high" or impact_score >= 70 or sentiment == "negative"

        if not needs_action:
            continue

        review_text = (analysis.get("review_text") or analysis.get("original_text") or "").strip()
        uploaded_at = analysis.get("uploaded_at")
        uploaded_order = uploaded_at.timestamp() if getattr(uploaded_at, "timestamp", None) else 0
        queue.append({
            "review_id": analysis.get("review_id"),
            "review_text": review_text[:160] + ("..." if len(review_text) > 160 else ""),
            "sentiment": sentiment,
            "urgency_level": urgency_level,
            "impact_score": impact_score,
            "experience_score": experience_score,
            "intent_label": (advanced.get("intent", {}) or {}).get("label") or "general_feedback",
            "priority": ((advanced.get("priority") or "investigate").replace("_", " ")).title(),
            "risk_flags": (advanced.get("risk_flags") or [])[:2],
            "top_aspects": list((analysis.get("aspect_sentiments") or {}).keys())[:3],
            "uploaded_at": uploaded_at,
            "_sort": (
                1 if urgency_level == "high" else 0,
                impact_score,
                1 if sentiment == "negative" else 0,
                uploaded_order,
            ),
        })

    queue.sort(key=lambda item: item["_sort"], reverse=True)
    for item in queue:
        item.pop("_sort", None)
    return queue[:limit]

# ---------- Home page ----------
@main.route("/")
def home():
    return render_template("home.html")

# ---------- User Login ----------
@main.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        cursor = dict_cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and check_password_hash(user["password_hash"], password):
            access_token = create_access_token(identity=str(user["user_id"]))
            response = redirect(url_for("main.dashboard"))
            set_access_cookies(response, access_token)

            # Flash after setting cookie
            session["_flashes"] = []
            flash("Login successful!", "success")
            return response

        session["_flashes"] = []
        flash("Invalid email or password.", "danger")
        return redirect(url_for("main.login"))

    return render_template("login.html")


# ---------- Admin Login ----------
from flask_jwt_extended import create_access_token, set_access_cookies

@main.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        cursor = dict_cursor()
        cursor.execute("SELECT * FROM admins WHERE username=%s", (username,))
        admin = cursor.fetchone()
        cursor.close()

        if admin and check_password_hash(admin["password_hash"], password):
            access_token = create_access_token(identity=admin["username"], additional_claims={"role": "admin"})
            resp = redirect(url_for("main.admin_dashboard"))
            set_access_cookies(resp, access_token)
            flash("Admin login successful!", "success")
            return resp

        flash("Invalid admin credentials.", "danger")
        return redirect(url_for("main.admin_login"))

    return render_template("admin_login.html")



# ---------- Register ----------
@main.route("/register", methods=["GET", "POST"])
def register():
    error_username = None
    error_email = None
    username = ""
    email = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("main.register"))

        password_error = password_rule_error(password, confirm_password)
        if password_error:
            flash(password_error, "danger")
            return render_template("register.html", username=username, email=email)

        cursor = dict_cursor()
        # unique check
        cursor.execute("SELECT user_id FROM users WHERE username=%s", (username,))
        if cursor.fetchone():
            error_username = "Username already exists"
        cursor.execute("SELECT user_id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            error_email = "Email already registered"

        if error_username or error_email:
            cursor.close()
            return render_template("register.html",
                                   error_username=error_username,
                                   error_email=error_email,
                                   username=username,
                                   email=email)

        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (%s, %s, %s, %s)",
            (username, email, password_hash, datetime.now(timezone.utc))
        )
        mysql.connection.commit()
        cursor.close()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("main.home"))

    return render_template("register.html")

# Admin Dashboard (User Details + Review Analysis)
from flask_jwt_extended import get_jwt
from flask import jsonify

@main.route("/admin_dashboard")
@jwt_required()
def admin_dashboard():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("SELECT user_id, username, email FROM users ORDER BY user_id")
    users = cursor.fetchall()

    for user in users:
        cursor.execute("""
            SELECT review_text, uploaded_at, overall_sentiment
            FROM reviews
            WHERE user_id=%s
            ORDER BY uploaded_at DESC LIMIT 2
        """, (user["user_id"],))
        user_reviews = cursor.fetchall()
        for r in user_reviews:
            r['highlighted_text'] = highlight_keywords(r['review_text'], r['overall_sentiment'])
        user["reviews"] = user_reviews

    cursor.execute("""
        SELECT r.review_id, r.review_text, r.uploaded_at, r.overall_sentiment,
               r.overall_sentiment_score, u.username
        FROM reviews r
        JOIN users u ON r.user_id = u.user_id
        ORDER BY r.uploaded_at DESC LIMIT 100
    """)
    reviews = cursor.fetchall()

    # Add highlighted text
    for r in reviews:
        r['highlighted_text'] = highlight_keywords(r['review_text'], r['overall_sentiment'])

    # Calculate sentiment counts for all reviews
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for r in reviews:
        sent = r['overall_sentiment'].lower() if r['overall_sentiment'] else 'neutral'
        if sent in sentiment_counts:
            sentiment_counts[sent] += 1

    cursor.close()

    return render_template("admin.html", users=users, reviews=reviews, sentiment_counts=sentiment_counts)

@main.route("/sentiment_trends")
@jwt_required()
def sentiment_trends():
    user_id = get_jwt_identity()
    claims = get_jwt()
    range_key, range_meta = parse_trend_range(request.args.get("range"))
    days = range_meta["days"]
    window_start = None
    if days is not None:
        window_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=max(days - 1, 0))

    base_where_clauses = []
    base_params = []
    if claims.get("role") != "admin":
        base_where_clauses.append("r.user_id = %s")
        base_params.append(user_id)

    series_where_clauses = list(base_where_clauses)
    series_params = list(base_params)
    if window_start is not None:
        series_where_clauses.append("r.uploaded_at >= %s")
        series_params.append(window_start)

    series_where_sql = f"WHERE {' AND '.join(series_where_clauses)}" if series_where_clauses else ""
    cursor = dict_cursor()
    cursor.execute(f"""
        SELECT
            DATE(r.uploaded_at) AS date,
            SUM(CASE WHEN LOWER(COALESCE(r.overall_sentiment, 'neutral')) = 'positive' THEN 1 ELSE 0 END) AS positive_count,
            SUM(CASE WHEN LOWER(COALESCE(r.overall_sentiment, 'neutral')) = 'negative' THEN 1 ELSE 0 END) AS negative_count,
            SUM(CASE WHEN LOWER(COALESCE(r.overall_sentiment, 'neutral')) = 'neutral' THEN 1 ELSE 0 END) AS neutral_count,
            COUNT(*) AS total_count,
            AVG(r.overall_sentiment_score) AS avg_score
        FROM reviews r
        {series_where_sql}
        GROUP BY DATE(r.uploaded_at)
        ORDER BY DATE(r.uploaded_at) ASC
    """, tuple(series_params))
    rows = cursor.fetchall()

    dates = [row['date'].strftime('%Y-%m-%d') for row in rows]
    positive_counts = [int(row['positive_count'] or 0) for row in rows]
    negative_counts = [int(row['negative_count'] or 0) for row in rows]
    neutral_counts = [int(row['neutral_count'] or 0) for row in rows]
    total_counts = [int(row['total_count'] or 0) for row in rows]

    current_summary = fetch_review_rollup(cursor, base_where_clauses, base_params, uploaded_after=window_start)
    current_total = int(current_summary.get("total_reviews") or 0)
    current_positive = int(current_summary.get("positive_reviews") or 0)
    current_negative = int(current_summary.get("negative_reviews") or 0)
    current_neutral = int(current_summary.get("neutral_reviews") or 0)
    current_avg_score = round(safe_float(current_summary.get("avg_sentiment_score")), 2)

    comparison_summary = None
    if days is not None and window_start is not None:
        previous_start = window_start - timedelta(days=days)
        previous_summary = fetch_review_rollup(
            cursor,
            base_where_clauses,
            base_params,
            uploaded_after=previous_start,
            uploaded_before=window_start,
        )
        previous_total = int(previous_summary.get("total_reviews") or 0)
        previous_positive = int(previous_summary.get("positive_reviews") or 0)
        previous_negative = int(previous_summary.get("negative_reviews") or 0)
        comparison_summary = {
            "label": f"Previous {days} days",
            "volume_delta": current_total - previous_total,
            "positive_delta": current_positive - previous_positive,
            "negative_delta": current_negative - previous_negative,
            "volume_delta_pct": round(((current_total - previous_total) / previous_total) * 100, 1) if previous_total else None,
        }

    cursor.close()

    daily_points = []
    for index, row in enumerate(rows):
        positive = positive_counts[index]
        negative = negative_counts[index]
        neutral = neutral_counts[index]
        total = total_counts[index]
        daily_points.append({
            "date": row["date"],
            "label": row["date"].strftime('%b %d'),
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "total_count": total,
            "balance": positive - negative,
        })

    risk_day = max(daily_points, key=lambda item: (item["negative_count"], item["total_count"]), default=None)
    best_day = max(daily_points, key=lambda item: (item["balance"], item["positive_count"]), default=None)
    notable_days = sorted(
        daily_points,
        key=lambda item: (item["negative_count"], item["total_count"], -item["balance"]),
        reverse=True,
    )[:5]

    trend_summary = {
        "total_reviews": current_total,
        "positive_share": round((current_positive / current_total) * 100, 1) if current_total else 0.0,
        "negative_share": round((current_negative / current_total) * 100, 1) if current_total else 0.0,
        "neutral_share": round((current_neutral / current_total) * 100, 1) if current_total else 0.0,
        "avg_daily_volume": round(current_total / len(daily_points), 1) if daily_points else 0.0,
        "avg_sentiment_score": current_avg_score,
    }

    return render_template(
        "sentiment_trends.html",
        dates=dates,
        positive_counts=positive_counts,
        negative_counts=negative_counts,
        neutral_counts=neutral_counts,
        total_counts=total_counts,
        selected_range=range_key,
        range_options=TREND_RANGE_OPTIONS,
        range_label=range_meta["label"],
        comparison_summary=comparison_summary,
        trend_summary=trend_summary,
        risk_day=risk_day,
        best_day=best_day,
        notable_days=notable_days,
    )




# ---------- Dashboard (Protected) ----------
@main.route("/dashboard")
@jwt_required()
def dashboard():
    user_id = get_jwt_identity()
    cursor = dict_cursor()
    cursor.execute("SELECT user_id, username, email FROM users WHERE user_id=%s", (user_id,))
    user = cursor.fetchone()
    cursor.execute("""
        SELECT
            COUNT(*) AS total_reviews,
            SUM(CASE WHEN LOWER(COALESCE(r.overall_sentiment, 'neutral')) = 'positive' THEN 1 ELSE 0 END) AS positive_reviews,
            SUM(CASE WHEN LOWER(COALESCE(r.overall_sentiment, 'neutral')) = 'negative' THEN 1 ELSE 0 END) AS negative_reviews,
            SUM(CASE WHEN LOWER(COALESCE(r.overall_sentiment, 'neutral')) = 'neutral' THEN 1 ELSE 0 END) AS neutral_reviews,
            AVG(r.overall_sentiment_score) AS avg_sentiment_score,
            MAX(r.uploaded_at) AS latest_upload
        FROM reviews r
        WHERE r.user_id = %s
    """, (user_id,))
    overview = cursor.fetchone() or {}

    cursor.execute("""
        SELECT
            SUM(CASE WHEN LOWER(COALESCE(ras.urgency_level, 'low')) = 'high' THEN 1 ELSE 0 END) AS high_urgency_reviews,
            AVG(ras.experience_score) AS avg_experience_score,
            AVG(ras.impact_score) AS avg_impact_score
        FROM reviews r
        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
        WHERE r.user_id = %s
    """, (user_id,))
    advanced_scores = cursor.fetchone() or {}

    cursor.execute("""
        SELECT LOWER(COALESCE(ras.intent_label, 'unknown')) AS label, COUNT(*) AS count
        FROM reviews r
        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
        WHERE r.user_id = %s
        GROUP BY 1
        ORDER BY count DESC, label ASC
    """, (user_id,))
    intent_breakdown = cursor.fetchall()

    cursor.execute("""
        SELECT LOWER(COALESCE(ras.language_code, 'unknown')) AS label, COUNT(*) AS count
        FROM reviews r
        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
        WHERE r.user_id = %s
        GROUP BY 1
        ORDER BY count DESC, label ASC
    """, (user_id,))
    language_breakdown = cursor.fetchall()

    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    current_window_start = now - timedelta(days=7)
    previous_window_start = now - timedelta(days=14)
    cursor.execute("""
        SELECT
            SUM(CASE WHEN uploaded_at >= %s THEN 1 ELSE 0 END) AS current_window,
            SUM(CASE WHEN uploaded_at >= %s AND uploaded_at < %s THEN 1 ELSE 0 END) AS previous_window
        FROM reviews
        WHERE user_id = %s
    """, (current_window_start, previous_window_start, current_window_start, user_id))
    cadence = cursor.fetchone() or {}
    cursor.close()

    analyses = fetch_review_analysis_payloads("WHERE r.user_id = %s", (user_id,), limit=80)
    issue_clusters = nlp_utils.cluster_reviews_by_similarity(analyses, max_clusters=4)
    aspect_trends = nlp_utils.compute_aspect_trends(analyses)
    priority_queue = build_priority_queue(analyses, limit=4)

    total_reviews = int(overview.get("total_reviews") or 0)
    positive_reviews = int(overview.get("positive_reviews") or 0)
    negative_reviews = int(overview.get("negative_reviews") or 0)
    neutral_reviews = int(overview.get("neutral_reviews") or 0)
    top_intent = intent_breakdown[0] if intent_breakdown else {"label": "unknown", "count": 0}
    dominant_language = language_breakdown[0] if language_breakdown else {"label": "unknown", "count": 0}
    current_window = int(cadence.get("current_window") or 0)
    previous_window = int(cadence.get("previous_window") or 0)

    dashboard_stats = {
        "total_reviews": total_reviews,
        "positive_reviews": positive_reviews,
        "negative_reviews": negative_reviews,
        "neutral_reviews": neutral_reviews,
        "positive_share": round((positive_reviews / total_reviews) * 100, 1) if total_reviews else 0.0,
        "avg_sentiment_score": round(safe_float(overview.get("avg_sentiment_score")), 2),
        "high_urgency_reviews": int(advanced_scores.get("high_urgency_reviews") or 0),
        "avg_experience_score": round(safe_float(advanced_scores.get("avg_experience_score")), 1),
        "avg_impact_score": round(safe_float(advanced_scores.get("avg_impact_score")), 1),
        "latest_upload": overview.get("latest_upload"),
        "top_intent": top_intent,
        "dominant_language": dominant_language,
        "review_velocity_current": current_window,
        "review_velocity_previous": previous_window,
        "review_velocity_delta": current_window - previous_window,
        "top_cluster": issue_clusters[0] if issue_clusters else None,
        "top_aspect_trend": aspect_trends[0] if aspect_trends else None,
    }

    return render_template(
        "dashboard.html",
        user=user,
        dashboard_stats=dashboard_stats,
        priority_queue=priority_queue,
        aspect_trends=aspect_trends[:6],
        issue_clusters=issue_clusters[:3],
        intent_breakdown=intent_breakdown[:4],
        language_breakdown=language_breakdown[:4],
    )


# ---------- Profile (view + update) ----------
# ---------- PROFILE ----------
@main.route("/profile", methods=["GET", "POST"])
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    cursor = dict_cursor()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Handle profile update
        if username or email:
            if not username or not email:
                flash("Username and email are required.", "warning")
            else:
                cursor.execute("SELECT user_id FROM users WHERE username=%s AND user_id<>%s", (username, user_id))
                username_conflict = cursor.fetchone()
                cursor.execute("SELECT user_id FROM users WHERE email=%s AND user_id<>%s", (email, user_id))
                email_conflict = cursor.fetchone()

                if username_conflict:
                    flash("Username already in use by another account.", "warning")
                if email_conflict:
                    flash("Email already in use by another account.", "warning")
                if not username_conflict and not email_conflict:
                    cursor.execute(
                        "UPDATE users SET username=%s, email=%s WHERE user_id=%s",
                        (username, email, user_id)
                    )
                    mysql.connection.commit()
                    flash("Profile updated successfully!", "success")

        # Handle password change
        password_change_requested = any([current_password, new_password, confirm_password])
        if password_change_requested:
            if not all([current_password, new_password, confirm_password]):
                flash("Fill in all password fields to change your password.", "warning")
            else:
                cursor.execute("SELECT password_hash FROM users WHERE user_id=%s", (user_id,))
                user_data = cursor.fetchone()
                password_error = password_rule_error(new_password, confirm_password)
                if not user_data or not check_password_hash(user_data["password_hash"], current_password):
                    flash("Current password is incorrect.", "danger")
                elif password_error:
                    flash(password_error, "danger")
                else:
                    new_password_hash = generate_password_hash(new_password)
                    cursor.execute(
                        "UPDATE users SET password_hash=%s WHERE user_id=%s",
                        (new_password_hash, user_id)
                    )
                    mysql.connection.commit()
                    flash("Password changed successfully!", "success")

    filters = parse_review_filters(request.args)
    where_sql, params = build_user_review_where_sql(user_id, filters)

    # fetch user
    cursor.execute("SELECT user_id, username, email FROM users WHERE user_id=%s", (user_id,))
    user = cursor.fetchone()

    cursor.execute("""
        SELECT
            SUM(CASE WHEN LOWER(COALESCE(overall_sentiment, 'neutral')) = 'positive' THEN 1 ELSE 0 END) AS positive_reviews,
            SUM(CASE WHEN LOWER(COALESCE(overall_sentiment, 'neutral')) = 'negative' THEN 1 ELSE 0 END) AS negative_reviews,
            SUM(CASE WHEN LOWER(COALESCE(overall_sentiment, 'neutral')) = 'neutral' THEN 1 ELSE 0 END) AS neutral_reviews
        FROM reviews
        WHERE user_id = %s
    """, (user_id,))
    portfolio_sentiment = cursor.fetchone() or {}

    cursor.execute(f"""
        SELECT
            r.review_id,
            r.review_text,
            r.uploaded_at,
            r.overall_sentiment,
            r.overall_sentiment_score,
            COALESCE(ras.intent_label, 'unknown') AS intent_label,
            COALESCE(ras.urgency_level, 'low') AS urgency_level,
            COALESCE(ras.impact_score, 0) AS impact_score,
            COALESCE(ras.language_code, 'unknown') AS language_code
        FROM reviews r
        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
        {where_sql}
        ORDER BY r.uploaded_at DESC
        LIMIT %s
    """, (*params, 80))
    reviews = cursor.fetchall()

    cursor.execute("""
        SELECT LOWER(COALESCE(ras.intent_label, 'unknown')) AS intent_label, COUNT(*) AS count
        FROM reviews r
        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
        WHERE r.user_id = %s
        GROUP BY 1
        ORDER BY count DESC, intent_label ASC
    """, (user_id,))
    intent_options = cursor.fetchall()
    cursor.close()

    sentiment_counts = {
        'positive': int(portfolio_sentiment.get('positive_reviews') or 0),
        'negative': int(portfolio_sentiment.get('negative_reviews') or 0),
        'neutral': int(portfolio_sentiment.get('neutral_reviews') or 0),
    }

    for review in reviews:
        review['highlighted_text'] = highlight_keywords(review['review_text'], review['overall_sentiment'])
        review['sentiment_key'] = normalize_sentiment(review.get('overall_sentiment'))

    analyses = fetch_review_analysis_payloads(where_sql, params, limit=80)
    issue_clusters = nlp_utils.cluster_reviews_by_similarity(analyses, max_clusters=4)
    aspect_trends = nlp_utils.compute_aspect_trends(analyses)
    language_breakdown = Counter(
        (analysis.get('advanced_insights', {}).get('language', {}).get('language') or 'unknown')
        for analysis in analyses
    )

    filtered_top_intent = Counter((review.get('intent_label') or 'unknown') for review in reviews).most_common(1)
    filter_summary = {
        "results_count": len(reviews),
        "high_urgency_count": sum(1 for review in reviews if (review.get("urgency_level") or "low").lower() == "high"),
        "avg_impact_score": round(
            sum(safe_float(review.get("impact_score")) for review in reviews) / len(reviews),
            1
        ) if reviews else 0.0,
        "avg_confidence": round(
            sum(safe_float(review.get("overall_sentiment_score")) for review in reviews) / len(reviews),
            2
        ) if reviews else 0.0,
        "top_intent": filtered_top_intent[0][0] if filtered_top_intent else "unknown",
        "has_active_filters": any([
            filters["q"],
            filters["sentiment"] != "all",
            filters["urgency"] != "all",
            filters["intent"] != "all",
        ]),
    }

    return render_template(
        "profile.html",
        user=user,
        reviews=reviews,
        sentiment_counts=sentiment_counts,
        issue_clusters=issue_clusters,
        aspect_trends=aspect_trends,
        language_breakdown=dict(language_breakdown),
        filters=filters,
        filter_summary=filter_summary,
        intent_options=intent_options,
    )


# ---------- Settings (Password Change) ----------
@main.route("/settings", methods=["GET", "POST"])
@jwt_required()
def settings():
    user_id = get_jwt_identity()
    cursor = dict_cursor()

    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Handle password change
        password_change_requested = any([current_password, new_password, confirm_password])
        if password_change_requested:
            if not all([current_password, new_password, confirm_password]):
                flash("Fill in all password fields to change your password.", "warning")
            else:
                cursor.execute("SELECT password_hash FROM users WHERE user_id=%s", (user_id,))
                user_data = cursor.fetchone()
                password_error = password_rule_error(new_password, confirm_password)
                if not user_data or not check_password_hash(user_data["password_hash"], current_password):
                    flash("Current password is incorrect.", "danger")
                elif password_error:
                    flash(password_error, "danger")
                else:
                    new_password_hash = generate_password_hash(new_password)
                    cursor.execute(
                        "UPDATE users SET password_hash=%s WHERE user_id=%s",
                        (new_password_hash, user_id)
                    )
                    mysql.connection.commit()
                    flash("Password changed successfully!", "success")
                    return redirect(url_for("main.settings"))

    # fetch user for display
    cursor.execute("SELECT user_id, username, email FROM users WHERE user_id=%s", (user_id,))
    user = cursor.fetchone()
    cursor.close()

    return render_template("settings.html", user=user)


# ---------- Upload Reviews (raw text) ----------
@main.route("/upload_review", methods=["GET", "POST"])
@jwt_required()
def upload_review():
    user_id = get_jwt_identity()
    if request.method == "POST":
        raw_review = (request.form.get("raw_review") or "").strip()
        file = request.files.get("file")
        rows = []

        # Case 1: raw text
        if raw_review:
            cursor = mysql.connection.cursor()
            sentiment_result = nlp_utils.enhanced_sentiment_analysis(raw_review)
            sentiment_label = sentiment_result['sentiment']
            sent_score = sentiment_result['confidence']
            cursor.execute("""
                INSERT INTO reviews (user_id, review_text, product_id, category, uploaded_at, overall_sentiment, overall_sentiment_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING review_id
            """, (user_id, raw_review, None, None, datetime.now(timezone.utc), sentiment_label, sent_score))
            review_id = cursor.fetchone()[0]
            mysql.connection.commit()
            cursor.close()
            nlp_utils.analyze_review_detailed(
                raw_review,
                sentiment_label,
                sent_score,
                mysql=mysql,
                review_id=review_id
            )
            alert_job = enqueue_analysis_job("alert_scan", str(user_id), {"scope": "recent"})
            enqueue_rq_analysis_job(current_app, alert_job["job_id"])
            flash("Review uploaded with sentiment!", "success")
            return redirect(url_for("main.profile"))

        # Case 2: CSV
        elif file and file.filename.lower().endswith(".csv"):
            stream = io.StringIO(file.stream.read().decode("utf-8"))
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "review_text" not in reader.fieldnames:
                flash("CSV must contain a 'review_text' column.", "danger")
                return redirect(url_for("main.upload_review"))
            for row in reader:
                text = (row.get("review_text") or "").strip()
                if text:
                    sentiment_result = nlp_utils.enhanced_sentiment_analysis(text)
                    sentiment_label = sentiment_result["sentiment"]
                    sent_score = sentiment_result["confidence"]
                    rows.append((user_id, text, None, None, datetime.now(timezone.utc), sentiment_label, sent_score))

            if rows:
                cursor = mysql.connection.cursor()
                inserted_reviews = []
                for row_values in rows:
                    cursor.execute("""
                        INSERT INTO reviews (user_id, review_text, product_id, category, uploaded_at, overall_sentiment, overall_sentiment_score)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                        RETURNING review_id, review_text, overall_sentiment, overall_sentiment_score
                    """, row_values)
                    inserted_reviews.append(cursor.fetchone())
                mysql.connection.commit()
                cursor.close()
                for review_id, review_text, overall_sentiment, overall_score in inserted_reviews:
                    nlp_utils.analyze_review_detailed(
                        review_text,
                        overall_sentiment,
                        overall_score or 0.0,
                        mysql=mysql,
                        review_id=review_id
                    )
                alert_job = enqueue_analysis_job("alert_scan", str(user_id), {"scope": "recent"})
                enqueue_rq_analysis_job(current_app, alert_job["job_id"])
                flash(f"Uploaded {len(rows)} review(s) with sentiment!", "success")
                return redirect(url_for("main.profile"))

        # If neither provided
        flash("Please provide raw review text or upload a CSV.", "warning")
        return redirect(url_for("main.upload_review"))

    # GET: show recent uploads for this user
    cursor = dict_cursor()
    cursor.execute("""
        SELECT
            r.review_id,
            r.review_text,
            r.uploaded_at,
            r.overall_sentiment,
            r.overall_sentiment_score,
            COALESCE(ras.intent_label, 'unknown') AS intent_label,
            COALESCE(ras.urgency_level, 'low') AS urgency_level,
            COALESCE(ras.impact_score, 0) AS impact_score
        FROM reviews r
        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
        WHERE r.user_id=%s
        ORDER BY r.uploaded_at DESC
        LIMIT 20
    """, (user_id,))
    reviews = cursor.fetchall()
    cursor.close()

    recent_count = len(reviews)
    positive_recent = sum(1 for review in reviews if normalize_sentiment(review.get("overall_sentiment")) == "positive")
    urgent_recent = sum(1 for review in reviews if (review.get("urgency_level") or "low").lower() == "high")
    dominant_intent = Counter((review.get("intent_label") or "unknown") for review in reviews).most_common(1)
    upload_summary = {
        "recent_count": recent_count,
        "positive_share": round((positive_recent / recent_count) * 100, 1) if recent_count else 0.0,
        "high_urgency_count": urgent_recent,
        "avg_confidence": round(
            sum(safe_float(review.get("overall_sentiment_score")) for review in reviews) / recent_count,
            2
        ) if recent_count else 0.0,
        "dominant_intent": dominant_intent[0][0] if dominant_intent else "unknown",
        "latest_upload": reviews[0]["uploaded_at"] if reviews else None,
    }
    return render_template("upload_reviews.html", reviews=reviews, upload_summary=upload_summary)


@main.route("/profile/export")
@jwt_required()
def export_reviews():
    user_id = get_jwt_identity()
    filters = parse_review_filters(request.args)
    where_sql, params = build_user_review_where_sql(user_id, filters)

    cursor = dict_cursor()
    cursor.execute(f"""
        SELECT
            r.review_id,
            r.uploaded_at,
            r.review_text,
            COALESCE(r.overall_sentiment, 'Neutral') AS overall_sentiment,
            COALESCE(r.overall_sentiment_score, 0) AS overall_sentiment_score,
            COALESCE(ras.intent_label, 'unknown') AS intent_label,
            COALESCE(ras.urgency_level, 'low') AS urgency_level,
            COALESCE(ras.experience_score, 0) AS experience_score,
            COALESCE(ras.impact_score, 0) AS impact_score,
            COALESCE(ras.language_code, 'unknown') AS language_code
        FROM reviews r
        LEFT JOIN review_aspect_sentiments ras ON ras.review_id = r.review_id
        {where_sql}
        ORDER BY r.uploaded_at DESC
    """, params)
    rows = cursor.fetchall()
    cursor.close()

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow([
        "review_id",
        "uploaded_at",
        "review_text",
        "overall_sentiment",
        "overall_sentiment_score",
        "intent_label",
        "urgency_level",
        "experience_score",
        "impact_score",
        "language_code",
    ])
    for row in rows:
        writer.writerow([
            row["review_id"],
            row["uploaded_at"].isoformat() if row.get("uploaded_at") else "",
            row["review_text"],
            row["overall_sentiment"],
            row["overall_sentiment_score"],
            row["intent_label"],
            row["urgency_level"],
            row["experience_score"],
            row["impact_score"],
            row["language_code"],
        ])

    export_data = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))
    export_data.seek(0)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return send_file(
        export_data,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"reviews-export-{timestamp}.csv",
    )


# -------- Delete Review --------
@main.route("/delete_review/<int:review_id>", methods=["POST"])
@jwt_required()
def delete_review(review_id):
    user_id = get_jwt_identity()
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM reviews WHERE review_id=%s AND user_id=%s", (review_id, user_id))
    deleted = cursor.rowcount
    mysql.connection.commit()
    cursor.close()

    if deleted:
        flash("Review deleted successfully!", "success")
    else:
        flash("Could not delete that review.", "warning")
    return redirect(url_for("main.profile"))



# ---------- Logout ----------
@main.route("/logout")
def logout():
    response = redirect(url_for("main.home"))
    unset_jwt_cookies(response)
    # flash("You have been logged out.", "info")
    return response

# ---------- Detailed Review Analysis ----------
@main.route("/review_analysis/<int:review_id>")
@jwt_required()
def review_analysis(review_id):
    """Return detailed analysis of a specific review."""
    user_id = get_jwt_identity()
    cursor = dict_cursor()

    # Fetch the review and verify ownership
    cursor.execute("""
        SELECT review_id, review_text, overall_sentiment, overall_sentiment_score
        FROM reviews
        WHERE review_id=%s AND user_id=%s
    """, (review_id, user_id))

    review = cursor.fetchone()
    cursor.close()

    if not review:
        return {"error": "Review not found or access denied"}, 404

    # Perform detailed analysis using NLP utilities
    analysis_result = nlp_utils.analyze_review_detailed(
        review['review_text'],
        review['overall_sentiment'],
        review['overall_sentiment_score'] or 0.0,
        mysql=mysql,
        review_id=review_id
    )

    return {
        'review_id': review_id,
        'original_text': analysis_result['original_text'],
        'highlighted_text': analysis_result['highlighted_text'],
        'aspects': analysis_result['aspects'],
        'aspect_sentiments': analysis_result['aspect_sentiments'],
        'summary': analysis_result['summary'],
        'advanced_insights': analysis_result['advanced_insights']
    }

# ---------- Admin API Routes ----------
@main.route("/admin/api/admin_stats", methods=["GET"])
@jwt_required()
def admin_api_stats():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("SELECT COUNT(*) as count FROM reviews")
    reviews_count = cursor.fetchone()['count']
    cursor.execute("SELECT COUNT(*) as count FROM aspect_categories")
    aspects_count = cursor.fetchone()['count']

    # Calculate accuracy as percentage of reviews with overall_sentiment_score >= 0.7
    cursor.execute("SELECT COUNT(*) as count FROM reviews WHERE overall_sentiment_score >= 0.7")
    accurate_reviews_count = cursor.fetchone()['count']
    accuracy = 0
    if reviews_count > 0:
        accuracy = round((accurate_reviews_count / reviews_count) * 100, 2)

    cursor.execute("SELECT COUNT(*) as count FROM review_aspect_sentiments WHERE urgency_level = 'high'")
    urgent_count = cursor.fetchone()['count']

    cursor.execute("SELECT AVG(experience_score) as avg_experience, AVG(impact_score) as avg_impact FROM review_aspect_sentiments")
    score_row = cursor.fetchone()

    cursor.close()

    return jsonify({
        "reviews": reviews_count,
        "aspects": aspects_count,
        "accuracy": accuracy,
        "urgent_reviews": urgent_count,
        "avg_experience": round(float(score_row['avg_experience'] or 0), 2),
        "avg_impact": round(float(score_row['avg_impact'] or 0), 2)
    })

# New endpoint for analytics data
@main.route("/admin/api/analytics_data", methods=["GET"])
@jwt_required()
def admin_api_analytics_data():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()

    # Get overall sentiment distribution counts
    cursor.execute("""
        SELECT overall_sentiment, COUNT(*) as count
        FROM reviews
        GROUP BY overall_sentiment
    """)
    sentiment_counts_raw = cursor.fetchall()
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for row in sentiment_counts_raw:
        sentiment = row['overall_sentiment'].lower() if row['overall_sentiment'] else 'neutral'
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] = row['count']

    # Get total count of all reviews
    cursor.execute("SELECT COUNT(*) as total FROM reviews")
    total_reviews = cursor.fetchone()['total']

    # Get recent reviews with username, sentiment, confidence (basic data only)
    cursor.execute("""
        SELECT r.review_id, r.review_text, r.overall_sentiment, r.overall_sentiment_score, u.username
        FROM reviews r
        JOIN users u ON r.user_id = u.user_id
        ORDER BY r.uploaded_at DESC
        LIMIT 10
    """)
    reviews = cursor.fetchall()
    # Convert datetime objects to strings for JSON serialization
    for review in reviews:
        if 'uploaded_at' in review and review['uploaded_at']:
            review['uploaded_at'] = review['uploaded_at'].isoformat()

    cursor.close()

    analyses = fetch_review_analysis_payloads(limit=80)
    clusters = nlp_utils.cluster_reviews_by_similarity(analyses, max_clusters=5)[:3]
    model_health = nlp_utils.get_model_health()

    return jsonify({
        "sentiment_counts": sentiment_counts,
        "reviews": reviews,
        "total_reviews": total_reviews,
        "issue_clusters": clusters,
        "model_health": model_health
    })


@main.route("/admin/api/issue_clusters", methods=["GET"])
@jwt_required()
def admin_api_issue_clusters():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    time_range = int(request.args.get('time_range', 30))
    analyses = fetch_review_analysis_payloads(
        "WHERE r.uploaded_at >= NOW() - (%s * INTERVAL '1 day')",
        (time_range,),
        limit=150
    )
    clusters = nlp_utils.cluster_reviews_by_similarity(analyses, max_clusters=6)
    return jsonify({"clusters": clusters})


@main.route("/admin/api/aspect_trends", methods=["GET"])
@jwt_required()
def admin_api_aspect_trends():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    time_range = int(request.args.get('time_range', 30))
    analyses = fetch_review_analysis_payloads(
        "WHERE r.uploaded_at >= NOW() - (%s * INTERVAL '1 day')",
        (time_range,),
        limit=160
    )
    trends = nlp_utils.compute_aspect_trends(analyses)
    return jsonify({"trends": trends})


@main.route("/admin/api/model_health", methods=["GET"])
@jwt_required()
def admin_api_model_health():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    health = nlp_utils.get_model_health()
    health["background_queue_ready"] = get_analysis_queue(current_app) is not None
    return jsonify(health)


@main.route("/admin/api/reanalysis_jobs", methods=["GET"])
@jwt_required()
def admin_api_reanalysis_jobs():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("""
        SELECT job_id, job_type, status, requested_by, created_at, started_at, completed_at, error_message
        FROM analysis_jobs
        ORDER BY created_at DESC
        LIMIT 20
    """)
    jobs = cursor.fetchall()
    cursor.close()
    return jsonify({"jobs": jobs})


@main.route("/admin/api/alerts", methods=["GET"])
@jwt_required()
def admin_api_alerts():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("""
        SELECT alert_id, alert_type, severity, title, message, status,
               delivered_slack, delivered_email, created_at, resolved_at
        FROM analysis_alerts
        ORDER BY created_at DESC
        LIMIT 20
    """)
    alerts = cursor.fetchall()
    cursor.close()
    return jsonify({"alerts": alerts})


@main.route("/admin/api/alerts/scan", methods=["POST"])
@jwt_required()
def admin_api_alert_scan():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    requested_by = str(get_jwt_identity())
    job = enqueue_analysis_job("alert_scan", requested_by, {"scope": "recent"})
    rq_job = enqueue_rq_analysis_job(current_app, job["job_id"])
    return jsonify({
        "message": "Alert scan queued",
        "job_id": job["job_id"],
        "execution_mode": "rq" if rq_job is not None else "database_queue"
    }), 202


@main.route("/admin/api/alerts/<int:alert_id>/resolve", methods=["POST"])
@jwt_required()
def admin_api_resolve_alert(alert_id):
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = mysql.connection.cursor()
    cursor.execute("""
        UPDATE analysis_alerts
        SET status = 'resolved', resolved_at = NOW()
        WHERE alert_id = %s AND status = 'open'
    """, (alert_id,))
    updated = cursor.rowcount
    mysql.connection.commit()
    cursor.close()
    if not updated:
        return jsonify({"error": "Alert not found or already resolved"}), 404
    return jsonify({"message": "Alert resolved"})


@main.route("/admin/api/reanalyze", methods=["POST"])
@jwt_required()
def admin_api_enqueue_reanalysis():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    payload = request.get_json(silent=True) or {}
    scope = payload.get("scope", "all")
    requested_by = str(get_jwt_identity())
    job = enqueue_analysis_job("reanalysis", requested_by, {"scope": scope})
    rq_job = enqueue_rq_analysis_job(current_app, job["job_id"])
    return jsonify({
        "message": "Re-analysis job queued",
        "job_id": job["job_id"],
        "status": job["status"],
        "execution_mode": "rq" if rq_job is not None else "database_queue"
    }), 202

# New endpoint for detailed review analysis (admin)
@main.route("/admin/api/review_analysis/<int:review_id>", methods=["GET"])
@jwt_required()
def admin_api_review_analysis(review_id):
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()

    # Fetch the review
    cursor.execute("""
        SELECT r.review_id, r.review_text, r.overall_sentiment, r.overall_sentiment_score, u.username
        FROM reviews r
        JOIN users u ON r.user_id = u.user_id
        WHERE r.review_id=%s
    """, (review_id,))

    review = cursor.fetchone()
    cursor.close()

    if not review:
        return {"error": "Review not found"}, 404

    # Perform detailed analysis using NLP utilities
    from .nlp_utils import analyze_review_detailed
    analysis_result = analyze_review_detailed(
        review['review_text'],
        review['overall_sentiment'],
        review['overall_sentiment_score'] or 0.0,
        mysql=mysql,
        review_id=review_id
    )

    return jsonify({
        'review_id': review_id,
        'username': review['username'],
        'original_text': analysis_result['original_text'],
        'clean_text': analysis_result['clean_text'],
        'highlighted_text': analysis_result['highlighted_text'],
        'aspects': analysis_result['aspects'],
        'aspect_sentiments': analysis_result['aspect_sentiments'],
        'summary': analysis_result['summary'],
        'advanced_insights': analysis_result['advanced_insights']
    })

@main.route("/admin/api/aspect_categories", methods=["GET"])
@jwt_required()
def admin_api_aspect_categories():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("SELECT id, name, description FROM aspect_categories ORDER BY name")
    categories = cursor.fetchall()
    cursor.close()

    return jsonify(categories)

@main.route("/admin/aspect_categories", methods=["POST"])
@jwt_required()
def admin_add_aspect_category():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()

    if not name:
        return {"error": "Name is required"}, 400

    cursor = mysql.connection.cursor()
    try:
        cursor.execute("INSERT INTO aspect_categories (name, description) VALUES (%s, %s)", (name, description))
        mysql.connection.commit()
        return {"message": "Aspect category added"}, 201
    except Exception as e:
        mysql.connection.rollback()
        return {"error": str(e)}, 500
    finally:
        cursor.close()

@main.route("/admin/api/sentiment_trends", methods=["GET"])
@jwt_required()
def admin_api_sentiment_trends():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    category = request.args.get('category', 'all')
    time_range = int(request.args.get('time_range', 30))
    sentiment_filter = request.args.get('sentiment', 'all')

    cursor = dict_cursor()

    # Build query
    query = """
        SELECT DATE(uploaded_at) as date,
               SUM(CASE WHEN overall_sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
               SUM(CASE WHEN overall_sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
               SUM(CASE WHEN overall_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count
        FROM reviews
        WHERE uploaded_at >= NOW() - (%s * INTERVAL '1 day')
    """
    params = [time_range]

    if category != 'all':
        query += " AND category = %s"
        params.append(category)

    if sentiment_filter != 'all':
        query += " AND overall_sentiment = %s"
        params.append(sentiment_filter)

    query += " GROUP BY DATE(uploaded_at) ORDER BY DATE(uploaded_at) ASC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()

    dates = [row['date'].strftime('%Y-%m-%d') for row in rows]
    positive = [row['positive_count'] for row in rows]
    negative = [row['negative_count'] for row in rows]
    neutral = [row['neutral_count'] for row in rows]

    return jsonify({"dates": dates, "positive": positive, "negative": negative, "neutral": neutral})

@main.route("/admin/api/aspect_sentiment_distribution", methods=["GET"])
@jwt_required()
def admin_api_aspect_sentiment_distribution():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    category = request.args.get('category', 'all')
    time_range = int(request.args.get('time_range', 30))
    sentiment_filter = request.args.get('sentiment', 'all')

    cursor = dict_cursor()

    # Fetch last 100 reviews for analysis
    query = """
        SELECT review_id, review_text, overall_sentiment
        FROM reviews
        WHERE uploaded_at >= NOW() - (%s * INTERVAL '1 day')
    """
    params = [time_range]

    if category != 'all':
        query += " AND category = %s"
        params.append(category)

    if sentiment_filter != 'all':
        query += " AND overall_sentiment = %s"
        params.append(sentiment_filter)

    query += " ORDER BY uploaded_at DESC LIMIT 100"

    cursor.execute(query, params)
    reviews = cursor.fetchall()
    cursor.close()

    # Analyze aspects and collect detailed data
    positive_aspects = {}
    negative_aspects = {}

    for review in reviews:
        analysis = nlp_utils.analyze_review_detailed(
            review['review_text'],
            review['overall_sentiment'],
            0.0,
            mysql=mysql,
            review_id=review['review_id']
        )
        for aspect, sent_info in analysis['aspect_sentiments'].items():
            sentiment = sent_info['sentiment']
            confidence = sent_info['confidence']

            if sentiment.lower() == 'positive':
                if aspect not in positive_aspects:
                    positive_aspects[aspect] = {'count': 0, 'total_confidence': 0.0}
                positive_aspects[aspect]['count'] += 1
                positive_aspects[aspect]['total_confidence'] += confidence
            elif sentiment.lower() == 'negative':
                if aspect not in negative_aspects:
                    negative_aspects[aspect] = {'count': 0, 'total_confidence': 0.0}
                negative_aspects[aspect]['count'] += 1
                negative_aspects[aspect]['total_confidence'] += confidence

    # Calculate average confidence and sort
    def process_aspects(aspects_dict):
        processed = []
        for aspect, data in aspects_dict.items():
            avg_confidence = data['total_confidence'] / data['count'] if data['count'] > 0 else 0.0
            processed.append({
                'aspect': aspect,
                'count': data['count'],
                'avg_confidence': round(avg_confidence, 2)
            })
        # Sort by count descending, then by avg_confidence descending
        processed.sort(key=lambda x: (x['count'], x['avg_confidence']), reverse=True)
        return processed[:10]  # Top 10

    top_positive = process_aspects(positive_aspects)
    top_negative = process_aspects(negative_aspects)

    return jsonify({
        'top_positive_aspects': top_positive,
        'top_negative_aspects': top_negative
    })

# --------- Admin API: Get all users ---------
@main.route("/admin/api/users", methods=["GET"])
@jwt_required()
def admin_api_get_users():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("""
        SELECT u.user_id, u.username, u.email, u.created_at, COUNT(r.review_id) as total_reviews
        FROM users u
        LEFT JOIN reviews r ON u.user_id = r.user_id
        GROUP BY u.user_id, u.username, u.email, u.created_at
        ORDER BY u.user_id
    """)
    users = cursor.fetchall()
    cursor.close()

    return jsonify({"users": users})

# --------- Admin API: Delete user ---------
@main.route("/admin/api/users/<int:user_id>", methods=["DELETE"])
@jwt_required()
def admin_api_delete_user(user_id):
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
    deleted = cursor.rowcount
    mysql.connection.commit()
    cursor.close()

    if deleted:
        return jsonify({"message": "User deleted successfully"})
    else:
        return jsonify({"error": "User not found"}), 404

@main.route("/admin/change_password", methods=["POST"])
@jwt_required()
def admin_change_password():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not current_password or not new_password or not confirm_password:
        return jsonify({"error": "All password fields are required."}), 400

    username = get_jwt_identity()  # admin username
    cursor = dict_cursor()
    cursor.execute("SELECT password_hash FROM admins WHERE username=%s", (username,))
    admin_data = cursor.fetchone()
    password_error = password_rule_error(new_password, confirm_password)

    if not admin_data or not check_password_hash(admin_data["password_hash"], current_password):
        cursor.close()
        return jsonify({"error": "Current password is incorrect."}), 400
    elif password_error:
        cursor.close()
        return jsonify({"error": password_error}), 400
    else:
        new_password_hash = generate_password_hash(new_password)
        cursor.execute("UPDATE admins SET password_hash=%s WHERE username=%s", (new_password_hash, username))
        mysql.connection.commit()
        cursor.close()
        return jsonify({"message": "Password changed successfully!"}), 200

@main.route("/admin/export_data", methods=["GET"])
@jwt_required()
def admin_export_data():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    format_type = request.args.get('format', 'excel').lower()

    cursor = dict_cursor()
    cursor.execute("""
        SELECT r.review_text, r.uploaded_at, r.overall_sentiment, r.overall_sentiment_score, u.username
        FROM reviews r
        JOIN users u ON r.user_id = u.user_id
        ORDER BY r.uploaded_at DESC
    """)
    reviews = cursor.fetchall()
    cursor.close()

    if format_type == 'pdf':
        # Generate PDF with detailed summary and pie chart
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.piecharts import Pie
        from reportlab.graphics import renderPDF
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from io import BytesIO

        bio = BytesIO()
        doc = SimpleDocTemplate(bio, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title = Paragraph("Review Summary Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        # Calculate sentiment counts
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for r in reviews:
            sent = r['overall_sentiment'].lower() if r['overall_sentiment'] else 'neutral'
            if sent in sentiment_counts:
                sentiment_counts[sent] += 1

        total_reviews = len(reviews)

        # Summary text
        summary_text = f"Total Reviews: {total_reviews}<br/>"
        if total_reviews > 0:
            summary_text += f"Positive: {sentiment_counts['positive']} ({sentiment_counts['positive']/total_reviews*100:.1f}%)<br/>"
            summary_text += f"Negative: {sentiment_counts['negative']} ({sentiment_counts['negative']/total_reviews*100:.1f}%)<br/>"
            summary_text += f"Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral']/total_reviews*100:.1f}%)<br/>"
        else:
            summary_text += "Positive: 0 (0.0%)<br/>"
            summary_text += "Negative: 0 (0.0%)<br/>"
            summary_text += "Neutral: 0 (0.0%)<br/>"

        # Determine dominant sentiment
        dominant = max(sentiment_counts, key=sentiment_counts.get)
        summary_text += f"<br/>Overall, the reviews show a predominantly <b>{dominant}</b> sentiment."

        summary = Paragraph(summary_text, styles['Normal'])
        story.append(summary)
        story.append(Spacer(1, 12))

        # Fetch top aspects (similar to analytics)
        cursor = dict_cursor()
        cursor.execute("""
            SELECT review_id, review_text, overall_sentiment
            FROM reviews
            ORDER BY uploaded_at DESC LIMIT 100
        """)
        recent_reviews = cursor.fetchall()
        cursor.close()

        # Analyze aspects
        positive_aspects = {}
        negative_aspects = {}

        for review in recent_reviews:
            analysis = nlp_utils.analyze_review_detailed(
                review['review_text'],
                review['overall_sentiment'],
                0.0,
                mysql=mysql,
                review_id=review['review_id']
            )
            for aspect, sent_info in analysis['aspect_sentiments'].items():
                sentiment = sent_info['sentiment']
                confidence = sent_info['confidence']

                if sentiment.lower() == 'positive':
                    if aspect not in positive_aspects:
                        positive_aspects[aspect] = {'count': 0, 'total_confidence': 0.0}
                    positive_aspects[aspect]['count'] += 1
                    positive_aspects[aspect]['total_confidence'] += confidence
                elif sentiment.lower() == 'negative':
                    if aspect not in negative_aspects:
                        negative_aspects[aspect] = {'count': 0, 'total_confidence': 0.0}
                    negative_aspects[aspect]['count'] += 1
                    negative_aspects[aspect]['total_confidence'] += confidence

        # Process top aspects
        def process_aspects(aspects_dict):
            processed = []
            for aspect, data in aspects_dict.items():
                avg_confidence = data['total_confidence'] / data['count'] if data['count'] > 0 else 0.0
                processed.append({
                    'aspect': aspect,
                    'count': data['count'],
                    'avg_confidence': round(avg_confidence, 2)
                })
            processed.sort(key=lambda x: (x['count'], x['avg_confidence']), reverse=True)
            return processed[:5]  # Top 5

        top_positive = process_aspects(positive_aspects)
        top_negative = process_aspects(negative_aspects)

        # Add top aspects to PDF
        if top_positive:
            pos_text = "<b>Top Positive Aspects:</b><br/>"
            for asp in top_positive:
                pos_text += f"- {asp['aspect']}: {asp['count']} mentions (avg confidence: {asp['avg_confidence']})<br/>"
            pos_para = Paragraph(pos_text, styles['Normal'])
            story.append(pos_para)
            story.append(Spacer(1, 6))

        if top_negative:
            neg_text = "<b>Top Negative Aspects:</b><br/>"
            for asp in top_negative:
                neg_text += f"- {asp['aspect']}: {asp['count']} mentions (avg confidence: {asp['avg_confidence']})<br/>"
            neg_para = Paragraph(neg_text, styles['Normal'])
            story.append(neg_para)
            story.append(Spacer(1, 12))

        # Pie chart
        if total_reviews > 0:
            drawing = Drawing(400, 200)
            pie = Pie()
            pie.x = 150
            pie.y = 50
            pie.width = 120
            pie.height = 120
            pie.data = [sentiment_counts['positive'], sentiment_counts['negative'], sentiment_counts['neutral']]
            pie.labels = ['Positive', 'Negative', 'Neutral']
            pie.slices.strokeWidth = 0.5
            pie.slices[0].fillColor = colors.green
            pie.slices[1].fillColor = colors.red
            pie.slices[2].fillColor = colors.yellow
            drawing.add(pie)
            story.append(drawing)

        # Bar chart for top positive aspects
        if top_positive:
            story.append(Spacer(1, 12))
            pos_title = Paragraph("<b>Top Positive Aspects</b>", styles['Heading2'])
            story.append(pos_title)
            story.append(Spacer(1, 6))

            drawing = Drawing(400, 200)
            bc = VerticalBarChart()
            bc.x = 50
            bc.y = 50
            bc.height = 125
            bc.width = 300
            bc.data = [[asp['count'] for asp in top_positive]]
            bc.categoryAxis.categoryNames = [asp['aspect'] for asp in top_positive]
            bc.valueAxis.valueMin = 0
            bc.bars[0].fillColor = colors.green
            drawing.add(bc)
            story.append(drawing)

        # Bar chart for top negative aspects
        if top_negative:
            story.append(Spacer(1, 12))
            neg_title = Paragraph("<b>Top Negative Aspects</b>", styles['Heading2'])
            story.append(neg_title)
            story.append(Spacer(1, 6))

            drawing = Drawing(400, 200)
            bc = VerticalBarChart()
            bc.x = 50
            bc.y = 50
            bc.height = 125
            bc.width = 300
            bc.data = [[asp['count'] for asp in top_negative]]
            bc.categoryAxis.categoryNames = [asp['aspect'] for asp in top_negative]
            bc.valueAxis.valueMin = 0
            bc.bars[0].fillColor = colors.red
            drawing.add(bc)
            story.append(drawing)

        doc.build(story)
        bio.seek(0)
        return send_file(bio, as_attachment=True, download_name='review_summary.pdf', mimetype='application/pdf')

    else:
        # Default to Excel
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.append(['Username', 'Review Text', 'Uploaded At', 'Sentiment', 'Score'])
        for r in reviews:
            ws.append([r['username'], r['review_text'], r['uploaded_at'], r['overall_sentiment'], r['overall_sentiment_score']])
        from io import BytesIO
        bio = BytesIO()
        wb.save(bio)
        bio.seek(0)
        return send_file(bio, as_attachment=True, download_name='reviews.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
