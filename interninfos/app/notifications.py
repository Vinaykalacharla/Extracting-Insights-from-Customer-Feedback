import json
import smtplib
from email.message import EmailMessage
from urllib.request import Request, urlopen


def send_slack_alert(webhook_url: str | None, title: str, message: str):
    if not webhook_url:
        return False
    payload = {"text": f"*{title}*\n{message}"}
    request = Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return 200 <= response.status < 300


def send_email_alert(config: dict, title: str, message: str):
    host = config.get("SMTP_HOST")
    to_email = config.get("ALERT_EMAIL_TO")
    from_email = config.get("SMTP_FROM_EMAIL")
    if not host or not to_email or not from_email:
        return False

    msg = EmailMessage()
    msg["Subject"] = title
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(message)

    port = int(config.get("SMTP_PORT") or 587)
    username = config.get("SMTP_USERNAME")
    password = config.get("SMTP_PASSWORD")

    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        if username and password:
            server.login(username, password)
        server.send_message(msg)
    return True
