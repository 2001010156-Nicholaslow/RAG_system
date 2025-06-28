# === Email Tool Wrapper ===
import logging
from typing import Optional
logger = logging.getLogger(__name__)

class EmailTool:
    name = "email"
    description = "Send or draft an email. Use only when user asks to contact someone or send a message."

    def __call__(self, action: str = "draft", message: str = "", email_address: Optional[str] = None) -> str:
        logger.info(f"Preparing email with action '{action}'")
        if action == "draft":
            return f"[DRAFT] Email Template:\n\n{message}"
        elif action == "send":
            if not email_address:
                return "❌ Email address is required to send email."
            # TODO: Add email sending logic (SMTP/API)
            return f"📤 Sent email to {email_address} with message:\n{message}"
        return "❌ Invalid action."
