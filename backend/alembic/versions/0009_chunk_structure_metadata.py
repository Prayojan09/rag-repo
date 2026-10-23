"""Persist section paths and chunk token counts."""

import sqlalchemy as sa

from alembic import op

revision = "0009_chunk_structure_metadata"
down_revision = "0008_chunk_embedding_context"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add optional structural metadata to existing and future chunks."""
    op.add_column("document_chunks", sa.Column("section_path", sa.JSON(), nullable=True))
    op.add_column("document_chunks", sa.Column("token_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Drop chunk section path and token count columns."""
    op.drop_column("document_chunks", "token_count")
    op.drop_column("document_chunks", "section_path")
