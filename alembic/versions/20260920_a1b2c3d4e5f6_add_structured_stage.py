"""add 'structured' value to job_stage enum

Revision ID: a1b2c3d4e5f6
Revises: eb3d0db311a0
Create Date: 2026-09-20
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "eb3d0db311a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ALTER TYPE ... ADD VALUE cannot run inside a transaction block on PG < 12; PG16 allows it.
    op.execute("ALTER TYPE job_stage ADD VALUE IF NOT EXISTS 'structured'")


def downgrade() -> None:
    # Postgres cannot drop an enum value; leaving it in place is harmless.
    pass
