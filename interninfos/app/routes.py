from flask import Blueprint, render_template, request, redirect, session, url_for, flash, current_app
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import csv
import io
import os
import MySQLdb.cursors

from . import mysql  # initialized in _init_.py
from . import nlp_utils  # Import NLP utilities

# Import specific functions from nlp_utils to avoid conflicts
from .nlp_utils import map_sentiment, highlight_keywords, preprocess_text

main = Blueprint('main', __name__, url_prefix="/")

# ---------- Helpers ----------
def dict_cursor():
    return mysql.connection.cursor(MySQLdb.cursors.DictCursor)

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

        if admin and admin["password_hash"] == password:
            access_token = create_access_token(identity=username, additional_claims={"role": "admin"})
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

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("main.register"))

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
            (username, email, password_hash, datetime.utcnow())
        )
        mysql.connection.commit()
        cursor.close()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("main.home"))

    return render_template("register.html")

# Admin Dashboard (User Details + Review Analysis)
from flask_jwt_extended import get_jwt

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




# ---------- Dashboard (Protected) ----------
@main.route("/dashboard")
@jwt_required()
def dashboard():
    user_id = get_jwt_identity()
    cursor = dict_cursor()
    cursor.execute("SELECT user_id, username, email FROM users WHERE user_id=%s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    return render_template("dashboard.html", user=user)


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

        cursor.execute("SELECT user_id FROM users WHERE email=%s AND user_id<>%s", (email, user_id))
        if cursor.fetchone():
            flash("Email already in use by another account.", "warning")
        else:
            cursor.execute(
                "UPDATE users SET username=%s, email=%s WHERE user_id=%s",
                (username, email, user_id)
            )
            mysql.connection.commit()
            flash("Profile updated successfully!", "success")

    # fetch user
    cursor.execute("SELECT user_id, username, email FROM users WHERE user_id=%s", (user_id,))
    user = cursor.fetchone()

    # fetch reviews WITH review_id
    cursor.execute("""
        SELECT review_id, review_text, uploaded_at, overall_sentiment, overall_sentiment_score
        FROM reviews
        WHERE user_id=%s
        ORDER BY uploaded_at DESC
        LIMIT 50
    """, (user_id,))
    reviews = cursor.fetchall()
    cursor.close()

    # Add highlighted text
    for r in reviews:
        r['highlighted_text'] = highlight_keywords(r['review_text'], r['overall_sentiment'])

    # Calculate sentiment counts for chart
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for r in reviews:
        sent = r['overall_sentiment'].lower() if r['overall_sentiment'] else 'neutral'
        if sent in sentiment_counts:
            sentiment_counts[sent] += 1

    return render_template("profile.html", user=user, reviews=reviews, sentiment_counts=sentiment_counts)



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
            # Analyze sentiment and irony
            sentiment_analyzer = nlp_utils.get_sentiment_analyzer()
            irony_analyzer = nlp_utils.get_irony_analyzer()
            sent_result = sentiment_analyzer(raw_review[:512])[0]
            sent_label, sent_score = sent_result["label"], float(sent_result["score"])
            irony_result = irony_analyzer(raw_review[:512])[0]
            irony_label, irony_score = irony_result["label"], float(irony_result["score"])
            sentiment_label = map_sentiment(sent_label, irony_label, irony_score)
            cursor.execute("""
                INSERT INTO reviews (user_id, review_text, product_id, category, uploaded_at, overall_sentiment, overall_sentiment_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, raw_review, None, None, datetime.utcnow(), sentiment_label, sent_score))
            mysql.connection.commit()
            cursor.close()
            flash("Review uploaded with sentiment!", "success")
            return redirect(url_for("main.profile"))

        # Case 2: CSV
        elif file and file.filename.lower().endswith(".csv"):
            stream = io.StringIO(file.stream.read().decode("utf-8"))
            reader = csv.DictReader(stream)
            if "review_text" not in reader.fieldnames:
                flash("CSV must contain a 'review_text' column.", "danger")
                return redirect(url_for("main.upload_review"))
            for row in reader:
                text = (row.get("review_text") or "").strip()
                if text:
                    # Analyze sentiment and irony
                    sentiment_analyzer = nlp_utils.get_sentiment_analyzer()
                    irony_analyzer = nlp_utils.get_irony_analyzer()
                    sent_result = sentiment_analyzer(text[:512])[0]
                    sent_label, sent_score = sent_result["label"], float(sent_result["score"])
                    irony_result = irony_analyzer(text[:512])[0]
                    irony_label, irony_score = irony_result["label"], float(irony_result["score"])
                    sentiment_label = map_sentiment(sent_label, irony_label, irony_score)
                    rows.append((user_id, text, None, None, datetime.utcnow(), sentiment_label, sent_score))

            if rows:
                cursor = mysql.connection.cursor()  # Define cursor here before use
                cursor.executemany("""
                    INSERT INTO reviews (user_id, review_text, product_id, category, uploaded_at, overall_sentiment, overall_sentiment_score)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                """, rows)
                mysql.connection.commit()
                cursor.close()
                flash(f"Uploaded {len(rows)} review(s) with sentiment!", "success")
                return redirect(url_for("main.profile"))

        # If neither provided
        flash("Please provide raw review text or upload a CSV.", "warning")
        return redirect(url_for("main.upload_review"))

    # GET: show recent uploads for this user
    cursor = dict_cursor()
    cursor.execute("""
        SELECT review_text, uploaded_at, overall_sentiment, overall_sentiment_score 
        FROM reviews WHERE user_id=%s ORDER BY uploaded_at DESC LIMIT 20
    """, (user_id,))
    reviews = cursor.fetchall()
    cursor.close()
    return render_template("upload_reviews.html", reviews=reviews)


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
        review['overall_sentiment_score'] or 0.0
    )

    return {
        'review_id': review_id,
        'original_text': analysis_result['original_text'],
        'highlighted_text': analysis_result['highlighted_text'],
        'aspects': analysis_result['aspects'],
        'aspect_sentiments': analysis_result['aspect_sentiments'],
        'summary': analysis_result['summary']
    }

