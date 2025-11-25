import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ================================
#  CONFIGURATION
# ================================
SMTP_SERVER = "smtp.hostinger.com"
SMTP_PORT = 465  # SSL port

EMAIL_ADDRESS = "admin@sahabs.tech"
EMAIL_PASSWORD = "Sahab@2025"

TO_EMAIL = "harunahk5575@gmail.com"  # any email you own to test

# ================================
#  EMAIL CONTENT
# ================================
subject = "Hostinger SMTP Test"
body = "Hello! This is a test email sent via Hostinger SMTP using Python."

msg = MIMEMultipart()
msg["From"] = EMAIL_ADDRESS
msg["To"] = TO_EMAIL
msg["Subject"] = subject
msg.attach(MIMEText(body, "plain"))

# ================================
#  SEND EMAIL
# ================================
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
