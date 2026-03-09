from flask import current_app, g

try:
    import psycopg2
except ModuleNotFoundError:
    psycopg2 = None


class PostgresCompat:
    """Minimal compatibility layer exposing `connection` like flask-mysqldb."""

    def init_app(self, app):
        app.teardown_appcontext(self.teardown)

    def _new_connection(self):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is required for database connections")

        database_url = current_app.config.get("DATABASE_URL")
        if database_url:
            return psycopg2.connect(database_url)

        return psycopg2.connect(
            host=current_app.config["DB_HOST"],
            port=current_app.config["DB_PORT"],
            user=current_app.config["DB_USER"],
            password=current_app.config["DB_PASSWORD"],
            dbname=current_app.config["DB_NAME"],
            sslmode=current_app.config.get("DB_SSLMODE", "require"),
        )

    @property
    def connection(self):
        conn = g.get("pg_conn")
        if conn is None or conn.closed:
            conn = self._new_connection()
            conn.autocommit = False
            g.pg_conn = conn
        return conn

    def teardown(self, error=None):
        conn = g.pop("pg_conn", None)
        if conn is None:
            return
        if error is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        conn.close()
