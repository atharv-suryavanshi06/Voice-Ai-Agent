"""
email_sender.py

Gmail SMTP Email Service for delivering recommended policy details and complete policy
documents to callers. Integrates with PostgreSQL to fetch policy document text and log delivery.
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Any, List, Optional

from core import config

logger = logging.getLogger(__name__)


def _safe_error_message(error: BaseException) -> str:
    """Return diagnostic context without credentials or recipient addresses."""
    message = str(error).replace(config.GMAIL_APP_PASSWORD or "__never__", "[redacted]")
    message = message.replace(config.GMAIL_SENDER_EMAIL or "__never__", "[redacted]")
    return f"{type(error).__name__}: {message[:300]}"


class EmailService:
    """Handles sending policy emails via Gmail SMTP and logging results to PostgreSQL."""

    def __init__(self) -> None:
        self.smtp_server = config.SMTP_SERVER
        self.smtp_port = config.SMTP_PORT
        self.sender_email = (config.GMAIL_SENDER_EMAIL or "").strip()
        self.app_password = (config.GMAIL_APP_PASSWORD or "").replace(" ", "").strip()


    def is_configured(self) -> bool:
        """Returns True if valid Gmail SMTP credentials are provided in config."""
        return bool(
            self.sender_email
            and self.app_password
            and self.sender_email != "your_email@gmail.com"
            and self.app_password != "your_16_char_app_password"
        )

    def send_policy_recommendation_email(
        self,
        recipient_email: str,
        customer_name: Optional[str],
        policies: List[Any],
        db_manager: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Sends an email containing recommended policies and full document text via Gmail SMTP.
        Logs delivery status to PostgreSQL.
        """
        if not recipient_email or "@" not in recipient_email:
            logger.warning("Invalid or absent recipient email")
            return False

        if not self.is_configured():
            msg = (
                "[Email Service Notice] Gmail SMTP is not configured in .env. "
                "Set GMAIL_SENDER_EMAIL and GMAIL_APP_PASSWORD to enable live email delivery."
            )
            print(msg)
            if db_manager and hasattr(db_manager, "log_sent_email"):
                db_manager.log_sent_email(
                    session_id=session_id or "unknown",
                    recipient_email=recipient_email,
                    policy_id=policies[0].policy_id if policies else "N/A",
                    policy_name=policies[0].policy_name if policies else "N/A",
                    status="SKIPPED_NOT_CONFIGURED",
                    error_message="Gmail SMTP credentials not set in .env",
                )
            return False

        name_str = customer_name.title() if customer_name else "Valued Customer"
        top_policy = policies[0] if policies else None
        top_policy_id = top_policy.policy_id if top_policy else "N/A"
        top_policy_name = top_policy.policy_name if top_policy else "Recommended Insurance Policy"

        # Fetch complete policy document text from PostgreSQL if database manager is available
        doc_data = None
        doc_text = ""
        pdf_path = None
        if db_manager and hasattr(db_manager, "get_policy_document") and top_policy:
            doc_data = db_manager.get_policy_document(top_policy.policy_id)
            if doc_data:
                doc_text = doc_data.get("document_text", "")
                pdf_path = doc_data.get("pdf_path")

        # Build Email Subject & Body
        subject = f"Your Suggested Insurance Policy Details - {top_policy_name}"

        # HTML Body
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; background-color: #f4f6f8; margin: 0; padding: 20px; }}
                .container {{ max-width: 680px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 10px; border: 1px solid #e0e0e0; }}
                .header {{ background-color: #0d47a1; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
                .header h2 {{ margin: 0; font-size: 24px; }}
                .policy-card {{ background: #f8fafc; border-left: 5px solid #0d47a1; padding: 15px; margin: 15px 0; border-radius: 4px; }}
                .policy-title {{ font-size: 18px; font-weight: bold; color: #0d47a1; margin-bottom: 5px; }}
                .meta-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                .meta-table td {{ padding: 8px; border-bottom: 1px solid #eee; }}
                .meta-table td.label {{ font-weight: bold; width: 40%; color: #555; }}
                .doc-box {{ background: #ffffff; border: 1px solid #ddd; padding: 15px; max-height: 400px; overflow-y: auto; font-family: monospace; white-space: pre-wrap; font-size: 13px; margin-top: 15px; border-radius: 5px; }}
                .footer {{ text-align: center; font-size: 12px; color: #888; margin-top: 30px; border-top: 1px solid #eee; padding-top: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Insurance Policy Summary</h2>
                </div>
                <p>Hi <strong>{name_str}</strong>,</p>
                <p>Thank you for speaking with our Voice AI advisor today! As discussed, here are the details of your recommended health insurance policy matched to your preferences:</p>
        """

        for p in policies:
            html_content += f"""
                <div class="policy-card">
                    <div class="policy-title">{p.policy_name} ({p.insurer})</div>
                    <table class="meta-table">
                        <tr><td class="label">Policy ID:</td><td>{p.policy_id}</td></tr>
                        <tr><td class="label">Plan Type:</td><td>{p.plan_type}</td></tr>
                        <tr><td class="label">Annual Premium:</td><td><strong>₹{p.premium:,.2f} / year</strong></td></tr>
                        <tr><td class="label">Sum Insured Coverage:</td><td><strong>₹{p.sum_insured:,.2f}</strong></td></tr>
                        <tr><td class="label">Pre-Existing Conditions:</td><td>Diabetes: {'Yes' if p.covers_diabetes else 'No'}, Hypertension: {'Yes' if p.covers_hypertension else 'No'}</td></tr>
                    </table>
                </div>
            """

        if doc_text:
            html_content += f"""
                <h3>Complete Policy Terms & Document Details</h3>
                <p>Below is the complete text of the policy document stored in our system database:</p>
                <div class="doc-box">{doc_text[:4000]} {"... (truncated for preview)" if len(doc_text) > 4000 else ""}</div>
            """

        html_content += """
                <p>If you have any questions or would like to complete your policy enrollment, please reply directly to this email or speak with your assigned advisor.</p>
                <div class="footer">
                    <p>Sent by Riya | Insurance Marketplace Voice Assistant</p>
                </div>
            </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = recipient_email

        # Attach HTML content
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        # Attach complete PDF document directly from PostgreSQL BYTEA or local disk
        pdf_bytes = doc_data.get("pdf_data") if doc_data else None
        pdf_name = f"{top_policy_name.replace(' ', '_')}_Policy_Document.pdf"

        if pdf_bytes:
            try:
                part = MIMEApplication(pdf_bytes, Name=pdf_name)
                part['Content-Disposition'] = f'attachment; filename="{pdf_name}"'
                msg.attach(part)
                print(f"[Email Service] Attached complete PDF ({len(pdf_bytes):,} bytes) fetched directly from PostgreSQL.")
            except Exception as pdf_err:
                logger.warning(f"Could not attach PostgreSQL PDF binary: {pdf_err}")
        elif pdf_path and os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(pdf_path))
                    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(pdf_path)}"'
                    msg.attach(part)
            except Exception as pdf_err:
                logger.warning(f"Could not attach PDF file: {pdf_err}")


        # Send via Gmail SMTP
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.app_password)
                server.sendmail(self.sender_email, [recipient_email], msg.as_string())

            print(f"[Email Service] Successfully sent policy recommendation email to '{recipient_email}'.")

            if db_manager and hasattr(db_manager, "log_sent_email"):
                db_manager.log_sent_email(
                    session_id=session_id or "unknown",
                    recipient_email=recipient_email,
                    policy_id=top_policy_id,
                    policy_name=top_policy_name,
                    status="SUCCESS",
                )
            return True
        except Exception as e:
            err_msg = _safe_error_message(e)
            print("[Email Service Error] Failed to send email via Gmail SMTP. See application logs for details.")
            logger.error("SMTP delivery failed: %s", err_msg)
            if db_manager and hasattr(db_manager, "log_sent_email"):
                db_manager.log_sent_email(
                    session_id=session_id or "unknown",
                    recipient_email=recipient_email,
                    policy_id=top_policy_id,
                    policy_name=top_policy_name,
                    status="FAILED",
                    error_message=err_msg,
                )
            return False
