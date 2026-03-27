"""Create a Google Form for Code of Conduct incident reports.

Builds the form via the Google Forms API v1, adding structured sections
for reporter info, incident details, description, evidence, and urgency.

Prerequisites
-------------
1. A Google Cloud project with the **Google Forms API** enabled.
2. OAuth 2.0 credentials (Desktop app) downloaded as ``credentials.json``
   in the working directory, **or** a service‑account key file whose path
   is set via the ``GOOGLE_APPLICATION_CREDENTIALS`` env var.
3. Install the client library::

       pip install google-api-python-client google-auth-oauthlib

Usage
-----
::

    python scripts/create_coc_report_form.py

The script prints the edit URL and the published (respondent) URL.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
]

TOKEN_PATH = Path("token.json")
CREDENTIALS_PATH = Path("credentials.json")

FORM_TITLE = "Code of Conduct Report"
FORM_DESCRIPTION = (
    "Use this form to report behaviour that may violate our Code of Conduct. "
    "All reports are treated confidentially and reviewed by the maintainer team. "
    "Fields marked with * are required."
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_credentials() -> Credentials:
    """Return valid OAuth 2.0 user credentials, refreshing or prompting as needed.

    Looks for a cached ``token.json`` first.  Falls back to the OAuth
    installed‑app flow using ``credentials.json``.

    Returns
    -------
    Credentials
        An authorized ``google.oauth2.credentials.Credentials`` instance.

    Raises
    ------
    FileNotFoundError
        If ``credentials.json`` is missing and no cached token exists.
    """
    creds: Credentials | None = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        if not CREDENTIALS_PATH.exists():
            raise FileNotFoundError(
                f"{CREDENTIALS_PATH} not found. Download OAuth 2.0 Desktop "
                "credentials from the Google Cloud Console."
            )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_PATH), SCOPES
        )
        creds = flow.run_local_server(port=0)

    TOKEN_PATH.write_text(json.dumps(json.loads(creds.to_json()), indent=2))
    return creds


# ---------------------------------------------------------------------------
# Form item builders
# ---------------------------------------------------------------------------

def _text_question(
    title: str,
    *,
    required: bool = False,
    paragraph: bool = False,
    description: str = "",
) -> dict[str, Any]:
    """Build a text‑input question item.

    Parameters
    ----------
    title:
        The question label shown to the respondent.
    required:
        Whether a response is mandatory.
    paragraph:
        ``True`` for a multi‑line text box; ``False`` for a single‑line input.
    description:
        Optional helper text displayed beneath the title.
    """
    item: dict[str, Any] = {
        "title": title,
        "questionItem": {
            "question": {
                "required": required,
                "textQuestion": {"paragraph": paragraph},
            }
        },
    }
    if description:
        item["description"] = description
    return item


def _choice_question(
    title: str,
    options: list[str],
    *,
    required: bool = False,
    choice_type: str = "RADIO",
    description: str = "",
) -> dict[str, Any]:
    """Build a choice (radio / checkbox / dropdown) question item.

    Parameters
    ----------
    title:
        The question label.
    options:
        List of choice strings.
    required:
        Whether a response is mandatory.
    choice_type:
        One of ``"RADIO"``, ``"CHECKBOX"``, or ``"DROP_DOWN"``.
    description:
        Optional helper text.
    """
    item: dict[str, Any] = {
        "title": title,
        "questionItem": {
            "question": {
                "required": required,
                "choiceQuestion": {
                    "type": choice_type,
                    "options": [{"value": v} for v in options],
                },
            }
        },
    }
    if description:
        item["description"] = description
    return item


def _page_break(title: str, description: str = "") -> dict[str, Any]:
    """Build a page‑break (section divider) item.

    Parameters
    ----------
    title:
        Section heading.
    description:
        Optional explanatory text for the section.
    """
    item: dict[str, Any] = {"title": title, "pageBreakItem": {}}
    if description:
        item["description"] = description
    return item


# ---------------------------------------------------------------------------
# Form definition
# ---------------------------------------------------------------------------

def build_form_items() -> list[dict[str, Any]]:
    """Return the ordered list of form items for the Code of Conduct report.

    Returns
    -------
    list[dict[str, Any]]
        Each element is a Google Forms API ``Item`` resource ready for a
        ``createItem`` batch‑update request.
    """
    items: list[dict[str, Any]] = []

    # -- Section 1: Reporter information ------------------------------------
    items.append(_text_question("Your name", description="Optional — you may report anonymously."))
    items.append(_text_question("Your contact email", description="Optional — only used if follow-up is needed."))
    items.append(_choice_question(
        "May we contact you for follow-up?",
        ["Yes", "No"],
    ))

    # -- Section 2: Incident details ----------------------------------------
    items.append(_page_break(
        "Incident Details",
        "Tell us who was involved and where the incident occurred.",
    ))
    items.append(_text_question("Person(s) involved", required=True))
    items.append(_text_question(
        "GitHub username(s) or other identifiers",
        description="If known, provide GitHub handles or other relevant identifiers.",
    ))
    items.append(_choice_question(
        "Where did this happen?",
        ["Issue", "Pull Request", "Discussion", "Chat", "Email", "Event", "Other"],
        required=True,
        choice_type="DROP_DOWN",
    ))
    items.append(_text_question(
        "Link(s) to the relevant content",
        paragraph=True,
        description="Paste URLs to issues, PRs, comments, messages, etc.",
    ))
    items.append(_text_question(
        "Date and time of incident",
        description="Approximate date/time and timezone, e.g. 2026-03-25 14:00 UTC.",
    ))

    # -- Section 3: Description ---------------------------------------------
    items.append(_page_break(
        "Description",
        "Please describe the incident in your own words.",
    ))
    items.append(_text_question(
        "Description of what happened",
        required=True,
        paragraph=True,
    ))
    items.append(_text_question(
        "Why are you concerned this may violate the Code of Conduct?",
        required=True,
        paragraph=True,
    ))

    # -- Section 4: Evidence & witnesses ------------------------------------
    items.append(_page_break(
        "Evidence & Witnesses",
        "Supporting materials help us investigate thoroughly.",
    ))
    items.append(_text_question(
        "Do you have screenshots, logs, or other evidence?",
        paragraph=True,
        description="Paste links to images, files, or Gists. You may also email attachments to the maintainers.",
    ))
    items.append(_text_question(
        "Were there any witnesses? If yes, list them.",
        paragraph=True,
    ))

    # -- Section 5: Urgency & support ---------------------------------------
    items.append(_page_break(
        "Urgency & Support",
    ))
    items.append(_choice_question(
        "Is there an immediate safety, harassment, or retaliation concern?",
        ["Yes", "No"],
        required=True,
    ))
    items.append(_text_question(
        "What outcome or support would be most helpful right now?",
        paragraph=True,
        description="Optional — let us know how we can best help.",
    ))

    # -- Section 6: Additional info -----------------------------------------
    items.append(_page_break("Additional Information"))
    items.append(_text_question(
        "Anything else we should know?",
        paragraph=True,
    ))

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_form() -> dict[str, Any]:
    """Authenticate, create the form, populate it, and return the Form resource.

    Returns
    -------
    dict[str, Any]
        The final Form resource as returned by the API (includes URLs and IDs).
    """
    creds = get_credentials()
    service = build("forms", "v1", credentials=creds)

    form = (
        service.forms()
        .create(body={"info": {"title": FORM_TITLE, "documentTitle": FORM_TITLE}})
        .execute()
    )
    form_id: str = form["formId"]

    update_body: dict[str, Any] = {
        "requests": [
            {
                "updateFormInfo": {
                    "info": {"description": FORM_DESCRIPTION},
                    "updateMask": "description",
                }
            }
        ]
    }

    items = build_form_items()
    for idx, item in enumerate(items):
        update_body["requests"].append({
            "createItem": {
                "item": item,
                "location": {"index": idx},
            }
        })

    service.forms().batchUpdate(formId=form_id, body=update_body).execute()

    form = service.forms().get(formId=form_id).execute()
    return form


def main() -> None:
    """Entry point — create the form and print its URLs."""
    form = create_form()

    form_id = form["formId"]
    edit_url = f"https://docs.google.com/forms/d/{form_id}/edit"
    respond_url = form.get("responderUri", f"https://docs.google.com/forms/d/e/{form_id}/viewform")

    print("Form created successfully!\n")
    print(f"  Title:        {form['info']['title']}")
    print(f"  Edit URL:     {edit_url}")
    print(f"  Respond URL:  {respond_url}")


if __name__ == "__main__":
    main()
