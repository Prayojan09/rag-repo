"""Store contextualized chunk text separately from clean citation text."""

import sqlalchemy as sa

from alembic import op

revision = "0008_chunk_embedding_context"
down_revision = "0007_document_quality_report"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add optional contextual text used for embedding and keyword retrieval."""
    op.add_column("document_chunks", sa.Column("embed_text", sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove contextual embedding text from stored chunks."""
    op.drop_column("document_chunks", "embed_text")
