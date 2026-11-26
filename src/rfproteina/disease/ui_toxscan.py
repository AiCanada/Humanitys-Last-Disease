"""Local picker controls for a user-initiated ToxScan website handoff."""

from __future__ import annotations

import base64
import binascii
from html import escape

from rfproteina.disease import toxscan_inputs

PREPARE = "results-toxscan-prepare"
IMPORT = "results-toxscan-import"
FILE = "toxscan_export_file"
TOKEN = "toxscan_request_token"


def encode_upload(payload: bytes) -> str:
    if not 0 < len(payload) <= toxscan_inputs.MAX_EXPORT_BYTES:
        return ""
    return base64.b64encode(payload).decode("ascii")


def decode_upload(fields: list[str]) -> bytes:
    if len(fields) != 1 or not 0 < len(fields[0]) <= 4 * ((toxscan_inputs.MAX_EXPORT_BYTES + 2) // 3):
        raise ValueError("Choose one original ToxScan CSV up to 2 MiB.")
    try:
        data = base64.b64decode(fields[0], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Choose the original ToxScan CSV file.") from error
    if len(data) > toxscan_inputs.MAX_EXPORT_BYTES:
        raise ValueError("Choose one original ToxScan CSV up to 2 MiB.")
    return data


def controls(candidate_selected: bool, request: dict | None) -> str:
    if not candidate_selected:
        return '<p class="hint">Choose one named ligand above to prepare a ToxScan website request.</p>'
    if request is None:
        return (f'<button type="submit" class="nav" name="__action" value="{PREPARE}" formnovalidate>'
                'Prepare a ToxScan website request</button>')
    return (
        '<fieldset><legend>Author-platform prediction</legend>'
        '<label>Canonical SMILES to copy<textarea readonly spellcheck="false" rows="2">'
        f'{escape(request["canonical_smiles"])}</textarea></label>'
        '<p>Submitting this structure on the ToxScan website shares it with the service operator. '
        'Download the original CSV when the prediction finishes.</p>'
        f'<p><a href="{escape(toxscan_inputs.SOURCE)}" target="_blank" rel="noopener noreferrer">'
        'Open ToxScan</a></p>'
        f'<input type="hidden" name="{TOKEN}" value="{escape(request["token"], quote=True)}">'
        f'<label>Downloaded CSV <input type="file" name="{FILE}" accept=".csv,text/csv"></label>'
        f'<button type="submit" class="nav" name="__action" value="{IMPORT}" formnovalidate>'
        'Verify and import prediction</button></fieldset>'
    )
