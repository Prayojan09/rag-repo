"""Store the optional formatted document preview separately from evidence text."""

import sqlalchemy as sa

from alembic import op

revision = "0006_structured_document_content"
down_revision = "0005_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("structured_content", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "structured_content")
