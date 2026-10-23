"""Persist ingestion quality diagnostics with each document."""

import sqlalchemy as sa

from alembic import op

revision = "0007_document_quality_report"
down_revision = "0006_structured_document_content"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the nullable report field for existing and future documents."""
    op.add_column("documents", sa.Column("quality_report", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Remove persisted ingestion report metadata."""
    op.drop_column("documents", "quality_report")
