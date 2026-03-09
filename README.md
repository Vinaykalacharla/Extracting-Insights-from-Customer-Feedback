# Extracting Insights from Customer Feedback

Flask application for collecting customer reviews, analyzing them with NLP, and surfacing user and admin insights from the same dataset.

The project combines a server-rendered Flask UI, PostgreSQL-backed persistence, JWT cookie authentication with CSRF protection, detailed review analysis, exports, and optional background jobs for reanalysis and alerting.

## Highlights

- User and admin authentication with JWT cookies and CSRF-protected writes
- Single-review upload and bulk CSV upload
- Overall sentiment plus aspect-level sentiment analysis
- Extra NLP signals: language, intent, urgency, impact, and experience scoring
- User dashboard with filters, detailed analysis modal, and CSV export
- Admin dashboard with trends, issue clustering, model health, alerts, and Excel/PDF export
- PostgreSQL-compatible storage, including hosted providers such as Supabase
- Optional Redis/RQ worker pipeline for reanalysis and alert scans
- Optional Slack and email alert delivery

## Tech Stack

- Backend: Flask, Jinja2, Werkzeug, Flask-JWT-Extended
- Database: PostgreSQL via `psycopg2`
- NLP: NLTK, spaCy, Hugging Face `transformers`, `sentence-transformers`, `langdetect`, `scikit-learn`
- Reporting and export: `pandas`, `openpyxl`, `xlsxwriter`, `reportlab`
- Background work: Redis + RQ
- Tests: `pytest`

## Repository Layout

```text
.
|-- README.md
`-- interninfos/
    |-- app.py
    |-- config.py
    |-- requirements.txt
    |-- schema.sql
    |-- db_migrations/
    |-- tests/
    |-- scripts/
    |   |-- create_admin.py
    |   |-- process_analysis_jobs.py
    |   |-- rq_worker.py
    |   `-- search_hf_models.py
    `-- app/
        |-- __init__.py
        |-- routes.py
        |-- nlp_utils.py
        |-- postgres.py
        |-- background.py
        |-- notifications.py
        |-- static/
        `-- templates/
```

## Core Product Flow

1. A user registers or signs in.
2. The user uploads one review or a CSV file containing a `review_text` column.
3. The app computes overall sentiment on insert and stores the review in PostgreSQL.
4. Detailed analysis is generated and cached in `review_aspect_sentiments`.
5. The user can inspect personal analytics and export filtered results as CSV.
6. Admin users can inspect broader analytics, run reanalysis jobs, scan for alerts, and export reports.

## Features

### User-facing

- Register, log in, and manage profile details
- Change password from the settings page
- Upload:
  - Raw review text
  - CSV files with a required `review_text` column
- Inspect:
  - Overall sentiment and confidence
  - Aspect-level sentiment
  - Highlighted review text
  - Intent, urgency, impact, and experience score
- Export filtered personal review history as CSV

### Admin-facing

- Separate admin login
- Review-level and aggregate analytics APIs
- Sentiment trends and aspect sentiment distribution
- Issue clustering and aspect trends
- Model health visibility
- Alert scan and reanalysis job tracking
- User management endpoints
- Export data as Excel or PDF summary reports

## Prerequisites

- Python 3 with `venv` and `pip`
- A PostgreSQL database
- Redis if you want asynchronous reanalysis and alert jobs
- Recommended for richer NLP:
  - spaCy English model: `en_core_web_sm`
  - locally available transformer models if network downloads are disabled

## Local Setup

Run all commands from the repository root.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Bash or zsh:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r interninfos/requirements.txt
```

### 3. Create the environment file

PowerShell:

```powershell
Copy-Item interninfos\.env.example interninfos\.env
```

Bash or zsh:

```bash
cp interninfos/.env.example interninfos/.env
```

### 4. Configure the app

Edit `interninfos/.env` and set at least:

```env
DATABASE_URL=postgresql://username:password@host:5432/postgres?sslmode=require
JWT_SECRET_KEY=replace-with-a-secure-secret
FLASK_SECRET_KEY=replace-with-a-secure-secret
MIN_PASSWORD_LENGTH=8
ALLOW_MODEL_DOWNLOADS=false
```

Notes:

- `DATABASE_URL` is the preferred database configuration.
- If `DATABASE_URL` is empty, the app falls back to `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, and `DB_SSLMODE`.
- Keep `JWT_COOKIE_CSRF_PROTECT=true` unless you have a very specific reason to disable it.
- Set `JWT_COOKIE_SECURE=true` when running behind HTTPS in production.

### 5. Initialize the database

Fresh database:

- Run [`interninfos/schema.sql`](interninfos/schema.sql).

Existing database that already has an older schema:

- Apply the migration files in [`interninfos/db_migrations/`](interninfos/db_migrations/) in numeric order.

### 6. Create an admin user

```bash
python interninfos/scripts/create_admin.py --username admin --email admin@example.com --password change-me
```

### 7. Run the app

```bash
python interninfos/app.py
```

Open `http://127.0.0.1:5000`.

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Recommended | Full PostgreSQL connection string |
| `DB_HOST` | Fallback only | Database host when `DATABASE_URL` is not set |
| `DB_PORT` | Fallback only | Database port |
| `DB_USER` | Fallback only | Database user |
| `DB_PASSWORD` | Fallback only | Database password |
| `DB_NAME` | Fallback only | Database name |
| `DB_SSLMODE` | Optional | PostgreSQL SSL mode, default `require` |
| `JWT_SECRET_KEY` | Yes | Secret used for JWT signing |
| `FLASK_SECRET_KEY` | Yes | Flask secret key |
| `MIN_PASSWORD_LENGTH` | Optional | Shared server-side password policy, default `8` |
| `ALLOW_MODEL_DOWNLOADS` | Optional | Allows transformer model downloads when `true` |
| `REDIS_URL` | Optional | Enables RQ background processing |
| `SLACK_WEBHOOK_URL` | Optional | Sends alert notifications to Slack |
| `SMTP_HOST` | Optional | SMTP server for email alerts |
| `SMTP_PORT` | Optional | SMTP server port, default `587` |
| `SMTP_USERNAME` | Optional | SMTP login username |
| `SMTP_PASSWORD` | Optional | SMTP login password |
| `SMTP_FROM_EMAIL` | Optional | Sender email address |
| `ALERT_EMAIL_TO` | Optional | Recipient for alert emails |
| `JWT_COOKIE_SECURE` | Production | Force secure cookies over HTTPS |
| `JWT_COOKIE_CSRF_PROTECT` | Optional | CSRF protection for JWT cookies, default `true` |

## Database Notes

The main tables are:

- `users`
- `admins`
- `reviews`
- `aspect_categories`
- `review_aspect_sentiments`
- `analysis_jobs`
- `analysis_alerts`

The application stores the original review in `reviews` and the richer cached analysis in `review_aspect_sentiments`. That cache is also used by admin analytics and export routes to avoid recomputing heavy NLP work on every request.

## NLP Behavior

The NLP layer is designed to degrade gracefully.

- Transformer and embedding model downloads are blocked by default unless `ALLOW_MODEL_DOWNLOADS=true`.
- If the required transformer models are not available locally, the app falls back instead of crashing.
- spaCy is loaded lazily. For better aspect extraction, install the English model:

```bash
python -m spacy download en_core_web_sm
```

- NLTK resources are fetched on demand if they are missing.

This makes local development easier, but for reliable production behavior you should preinstall the required models and corpora in your runtime environment.

## Background Jobs and Alerts

The app works without Redis, but asynchronous jobs will not be processed automatically.

If you want queued reanalysis and alert scanning:

1. Set `REDIS_URL`.
2. Start a Redis server.
3. Run the RQ worker:

```bash
python interninfos/scripts/rq_worker.py
```

If you only want to process one queued job manually:

```bash
python interninfos/scripts/process_analysis_jobs.py
```

Alert delivery is optional:

- Slack alerts require `SLACK_WEBHOOK_URL`
- Email alerts require `SMTP_HOST`, `SMTP_FROM_EMAIL`, `ALERT_EMAIL_TO`, and usually SMTP credentials

## Useful Scripts

| Script | Purpose |
| --- | --- |
| `interninfos/scripts/create_admin.py` | Create or update an admin account |
| `interninfos/scripts/rq_worker.py` | Start an RQ worker for queued analysis jobs |
| `interninfos/scripts/process_analysis_jobs.py` | Process one queued analysis job synchronously |
| `interninfos/scripts/search_hf_models.py` | Explore Hugging Face model options |

## Testing

Run the NLP test suite:

```bash
python -m pytest interninfos/tests/test_nlp.py -q
```

If you want the tests to avoid model downloads explicitly:

PowerShell:

```powershell
$env:ALLOW_MODEL_DOWNLOADS="false"
python -m pytest interninfos/tests/test_nlp.py -q
```

Bash or zsh:

```bash
ALLOW_MODEL_DOWNLOADS=false python -m pytest interninfos/tests/test_nlp.py -q
```

## Deployment Notes

This repository is ready for staging, but production deployment still needs normal hardening steps.

- [`interninfos/app.py`](interninfos/app.py) runs the Flask development server and should not be your final production process manager.
- Put the Flask app behind a real WSGI server in production.
- Use PostgreSQL with SSL enabled.
- Set strong secrets for `JWT_SECRET_KEY` and `FLASK_SECRET_KEY`.
- Set `JWT_COOKIE_SECURE=true` under HTTPS.
- Run Redis plus the worker if admin queue features must operate continuously.
- Preinstall NLP models if your deployment environment blocks outbound downloads.

## Troubleshooting

### App cannot connect to the database

- Verify `DATABASE_URL` first.
- If you use the fallback `DB_*` fields, confirm all values are set and reachable.
- For Supabase or hosted Postgres, keep `sslmode=require`.

### Background jobs stay queued

- Confirm `REDIS_URL` is set.
- Confirm Redis is running.
- Confirm `python interninfos/scripts/rq_worker.py` is running in a separate process.

### NLP results are limited

- Install `en_core_web_sm`.
- Allow model downloads temporarily or preinstall the required Hugging Face models.
- Check the admin model health endpoint in the dashboard.

### Form submissions fail unexpectedly

- Make sure you are using the app on the same origin as the cookie-based auth flow.
- Keep JWT cookie CSRF protection enabled and do not strip cookies or CSRF headers in front-end proxies.

## Security Notes

- Do not commit `interninfos/.env`.
- Use strong random secrets for JWT and Flask sessions.
- Keep CSRF protection enabled for authenticated writes.
- Use secure cookies in production.
- Review outbound alert destinations before enabling Slack or email delivery.
