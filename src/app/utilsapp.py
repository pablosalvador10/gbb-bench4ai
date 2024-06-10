import black
import os
import smtplib
from typing import List
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from utils.ml_logging import get_logger
import markdown

# Set up logger
logger = get_logger()

# Function to save uploaded file and return path (synchronously for now)
def save_uploaded_file(uploaded_file):
    os.makedirs("tempDir", exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join("tempDir", uploaded_file.name)
    with open(file_path, "wb") as f:
        contents = uploaded_file.read()
        f.write(contents)
    return file_path


def process_code(text: str) -> str:
    try:
        # Format the code using the 'black' formatter
        formatted_code = black.format_str(text, mode=black.FileMode())
        return formatted_code
    except black.InvalidInput as e:
        return f"Invalid input: {e}"


def send_email(
    response: str,
    from_email: str,
    to_emails: List[str],
    cc_emails: Optional[List[str]] = None,
    subject: Optional[str] = "Submission Response",
) -> None:
    """
    Send an email with the given response.

    Args:
        response (str): The response to send in the email.
        from_email (str): The email address to send the email from.
        to_emails (List[str]): The email addresses to send the email to.
        cc_emails (Optional[List[str]], optional): The email addresses to CC on the email. Defaults to None.
        subject (Optional[str], optional): The subject of the email. Defaults to "Submission Response".
    """
    # create message object instance
    msg = MIMEMultipart()

    # setup the parameters of the message
    password = os.getenv("EMAIL_PASSWORD")
    msg["From"] = from_email
    to_emails = [email for sublist in to_emails for email in sublist]
    msg["To"] = ", ".join(to_emails)
    if cc_emails:
        msg["Cc"] = ", ".join(cc_emails)
    msg["Subject"] = subject

    logger.info("From: %s", msg["From"])
    logger.info("To: %s", msg["To"])
    if cc_emails:
        logger.info("Cc: %s", msg["Cc"])
    logger.info("Subject: %s", msg["Subject"])

    html = markdown.markdown(response)

    msg.attach(MIMEText(html, "html"))

    logger.info("Message body: %s", response)

    server = smtplib.SMTP("smtp.office365.com: 587")

    server.starttls()

    server.login(msg["From"], password)

    # Send the email to the 'To' and 'Cc' addresses
    server.sendmail(msg["From"], to_emails + (cc_emails if cc_emails else []), msg.as_string())

    server.quit()

    logger.info("Email sent to %s", msg["To"])