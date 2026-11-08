"""Check archived byte hashes, with an explicit CSV line-ending migration.

The supplied November 3 project has LF CSV files while several original
manifests record CRLF bytes. Only a pure LF-to-CRLF reconstruction that exactly
matches the archived SHA-256 is accepted. No numeric/text bytes are normalized;
non-CSV files and mixed/bare-CR files still require exact bytes. The current
release manifest always records the actual supplied bytes.
"""
from pathlib import Path
import hashlib

CSV_LINE_ENDING_MATCHES = {}


def check_archived_hash(path, expected, project=None):
    path = Path(path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual == expected:
        return
    if path.suffix.lower() == '.csv' and b'\r' not in raw:
        restored = raw.replace(b'\n', b'\r\n')
        if hashlib.sha256(restored).hexdigest() == expected:
            relative = str(path.relative_to(project)) if project else str(path)
            CSV_LINE_ENDING_MATCHES[relative] = {
                'supplied_bytes_sha256': actual, 'archived_crlf_sha256': expected,
                'operation': 'LF -> CRLF only; all other bytes unchanged',
                'line_feed_count': raw.count(b'\n'),
            }
            return
    raise AssertionError(f'Archived hash mismatch: {path}')
