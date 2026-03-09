import argparse
import os

import psycopg2
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash


def get_connection():
    load_dotenv("interninfos/.env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing in interninfos/.env")
    return psycopg2.connect(database_url)


def main():
    parser = argparse.ArgumentParser(description="Create or update an admin user.")
    parser.add_argument("--username", required=True, help="Admin username")
    parser.add_argument("--email", required=True, help="Admin email")
    parser.add_argument("--password", required=True, help="Admin password")
    args = parser.parse_args()

    password_hash = generate_password_hash(args.password)

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO admins (username, email, password_hash)
            VALUES (%s, %s, %s)
            ON CONFLICT (username)
            DO UPDATE SET
                email = EXCLUDED.email,
                password_hash = EXCLUDED.password_hash
            """,
            (args.username.strip(), args.email.strip().lower(), password_hash),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    print("ADMIN_UPSERT_OK")


if __name__ == "__main__":
    main()
