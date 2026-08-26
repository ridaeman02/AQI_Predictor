import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_aqi_email(city, aqi, category, timestamp, recommendation, threshold):
    """
    Sends an SMTP-based email alert for hazardous AQI levels.
    """
    enabled = os.getenv("ALERT_EMAIL_ENABLED", "false").strip().lower() == "true"
    if not enabled:
        print("AQI Alerts: Email notifications are disabled.")
        return

    recipient = os.getenv("ALERT_EMAIL_TO")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port_raw = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USERNAME")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    # Validation of required configs
    missing = []
    if not recipient: missing.append("ALERT_EMAIL_TO")
    if not smtp_host: missing.append("SMTP_HOST")
    if not smtp_port_raw: missing.append("SMTP_PORT")
    if not smtp_user: missing.append("SMTP_USERNAME")
    if not smtp_pass: missing.append("SMTP_PASSWORD")

    if missing:
        err_msg = f"AQI Alert configuration error: Missing environment variables {missing}."
        print(err_msg)
        raise ValueError(err_msg)

    recipient = recipient.strip()
    smtp_host = smtp_host.strip()
    smtp_user = smtp_user.strip()
    smtp_pass = smtp_pass.strip()
    
    try:
        smtp_port = int(smtp_port_raw.strip())
    except ValueError:
        raise ValueError(f"SMTP_PORT must be a valid integer, got: {smtp_port_raw}")

    # Build Email Message
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg["Subject"] = f"AQI Alert — {city}"

    body = (
        f"AQI Alert — {city}\n\n"
        f"City: {city}\n"
        f"Predicted AQI: {aqi:.2f}\n"
        f"Category: {category}\n"
        f"Alert threshold: {threshold}\n"
        f"Prediction time: {timestamp}\n\n"
        f"Recommendation:\n{recommendation}\n"
    )
    msg.attach(MIMEText(body, "plain"))

    try:
        # SMTP Session Initiation
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipient, msg.as_string())
        server.quit()
        print(f"AQI Alert Email successfully sent to {recipient} for {city}.")
    except Exception as e:
        err_msg = f"AQI alert email delivery failed: {str(e)}"
        # Print warning but raise it to ensure it is visible in actions logs
        print(err_msg)
        raise RuntimeError(err_msg) from e
