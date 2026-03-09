-- Supabase/PostgreSQL schema
-- Run this in the Supabase SQL Editor (or psql) for your target database.

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    review_text TEXT NOT NULL,
    product_id VARCHAR(100) NULL,
    category VARCHAR(100),
    uploaded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    overall_sentiment VARCHAR(50),
    overall_sentiment_score DOUBLE PRECISION,
    CONSTRAINT fk_reviews_user
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS admins (
    admin_id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS aspect_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_aspect_sentiments (
    review_id INT PRIMARY KEY,
    aspect_sentiments JSONB NOT NULL,
    analysis_payload JSONB,
    language_code VARCHAR(20),
    language_confidence DOUBLE PRECISION,
    intent_label VARCHAR(50),
    urgency_level VARCHAR(20),
    experience_score INT,
    impact_score DOUBLE PRECISION,
    cached_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cache_review
        FOREIGN KEY (review_id) REFERENCES reviews(review_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analysis_jobs (
    job_id SERIAL PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    requested_by VARCHAR(100),
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS analysis_alerts (
    alert_id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    payload JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    delivered_slack BOOLEAN NOT NULL DEFAULT FALSE,
    delivered_email BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_reviews_uploaded_at ON reviews (uploaded_at);
CREATE INDEX IF NOT EXISTS idx_reviews_category ON reviews (category);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews (overall_sentiment);
CREATE INDEX IF NOT EXISTS idx_cached_at ON review_aspect_sentiments (cached_at);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs (status, created_at);
CREATE INDEX IF NOT EXISTS idx_analysis_alerts_status ON analysis_alerts (status, created_at);
