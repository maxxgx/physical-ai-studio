"""normalize uvc fingerprints

Strip trailing :N suffix from old FrameSource-style UVC fingerprints.
Examples: /dev/video0:0 -> /dev/video0, FaceTime HD Camera:0 -> FaceTime HD Camera

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-28 00:00:00.000000

"""

import re
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SUFFIX_RE = re.compile(r":\d+$")


def upgrade() -> None:
    """Strip trailing :N suffix from old FrameSource UVC fingerprints."""
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT id, fingerprint FROM project_cameras WHERE driver = 'usb_camera'")).fetchall()

    for row in rows:
        camera_id, fingerprint = row
        normalized = _SUFFIX_RE.sub("", fingerprint)
        if normalized != fingerprint:
            conn.execute(
                sa.text("UPDATE project_cameras SET fingerprint = :fp WHERE id = :id"),
                {"fp": normalized, "id": str(camera_id)},
            )


def downgrade() -> None:
    """Not reversible — old suffixed fingerprints are not recoverable."""
