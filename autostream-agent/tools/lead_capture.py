"""
tools/lead_capture.py
Mock lead capture tool for AutoStream agent.
"""

import json
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API call to capture a qualified lead.
    In production this would POST to a CRM or backend endpoint.

    Args:
        name     : Full name of the prospective user
        email    : Email address
        platform : Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict with status and confirmation details
    """
    # Basic validation
    if not all([name, email, platform]):
        return {"status": "error", "message": "Missing required fields."}

    if "@" not in email or "." not in email.split("@")[-1]:
        return {"status": "error", "message": "Invalid email address."}

    timestamp = datetime.utcnow().isoformat() + "Z"

    # Simulate successful CRM write
    lead_record = {
        "lead_id": f"AS-{abs(hash(email)) % 100000:05d}",
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": timestamp,
        "source": "Inflx-AutoStream-Agent",
        "status": "new",
    }

    # --- Required print (as specified in assignment) ---
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    # ---------------------------------------------------

    return {"status": "success", "lead": lead_record}
