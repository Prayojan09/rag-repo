"""Add tenant ownership to documents and feedback records."""

import sqlalchemy as sa

from alembic import op

revision = "0004_tenant_isolation"
down_revision = "0003_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("tenant_id", sa.String(length=64), nullable=True, server_default="default"),
    )
    op.add_column(
        "chat_feedback",
        sa.Column("tenant_id", sa.String(length=64), nullable=True, server_default="default"),
    )
    op.alter_column("documents", "tenant_id", nullable=False)
    op.alter_column("chat_feedback", "tenant_id", nullable=False)

    op.drop_index("ix_documents_sha256", table_name="documents")
    op.drop_constraint("uq_document_filename_version", "documents", type_="unique")
    op.create_unique_constraint("uq_document_tenant_sha256", "documents", ["tenant_id", "sha256"])
    op.create_unique_constraint(
        "uq_document_tenant_filename_version",
        "documents",
        ["tenant_id", "filename", "version"],
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])
    op.create_index("ix_chat_feedback_tenant_id", "chat_feedback", ["tenant_id"])
    op.create_index("ix_documents_sha256", "documents", ["sha256"])


def downgrade() -> None:
    op.drop_index("ix_documents_sha256", table_name="documents")
    op.drop_index("ix_chat_feedback_tenant_id", table_name="chat_feedback")
    op.drop_index("ix_documents_tenant_id", table_name="documents")
    op.drop_constraint("uq_document_tenant_filename_version", "documents", type_="unique")
    op.drop_constraint("uq_document_tenant_sha256", "documents", type_="unique")
    op.create_unique_constraint(
        "uq_document_filename_version", "documents", ["filename", "version"]
    )
    op.create_index("ix_documents_sha256", "documents", ["sha256"], unique=True)
    op.drop_column("chat_feedback", "tenant_id")
    op.drop_column("documents", "tenant_id")
