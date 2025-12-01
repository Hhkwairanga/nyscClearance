import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

"""
Minimal SMTP test script.

Reads all configuration from environment variables to avoid hardcoding credentials.

Required env vars:
  - SMTP_SERVER (e.g., smtp.hostinger.com)
  - SMTP_PORT (e.g., 465 for SSL)
  - EMAIL_ADDRESS (from address)
  - EMAIL_PASSWORD (SMTP password)
  - TO_EMAIL (recipient email for testing)
"""

SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
TO_EMAIL = os.getenv("TO_EMAIL", "")

subject = os.getenv("SMTP_SUBJECT", "Hostinger SMTP Test")
body = os.getenv("SMTP_BODY", "Hello! This is a test email sent via SMTP using Python.")

if not all([SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD, TO_EMAIL]):
    raise SystemExit("Missing one or more required env vars: SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD, TO_EMAIL")

msg = MIMEMultipart()
msg["From"] = EMAIL_ADDRESS
msg["To"] = TO_EMAIL
msg["Subject"] = subject
msg.attach(MIMEText(body, "plain"))

try:
    print("Connecting to SMTP server...")
    server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
    print("Logging in...")
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    print("Sending email...")
    server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
    print("✔ Email sent successfully!")
    server.quit()
except Exception as e:
    print("❌ Failed to send email")
    print("Error:", str(e))
