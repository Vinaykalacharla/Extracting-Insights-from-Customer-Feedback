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

CREATE INDEX IF NOT EXISTS idx_analysis_alerts_status ON analysis_alerts (status, created_at);
