-- Migration script to create review_aspect_sentiments cache table
CREATE TABLE IF NOT EXISTS review_aspect_sentiments (
    review_id INT PRIMARY KEY,
    aspect_sentiments JSONB NOT NULL,
    cached_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cache_review FOREIGN KEY (review_id) REFERENCES reviews(review_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cached_at ON review_aspect_sentiments (cached_at);
