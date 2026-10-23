"""Store answer feedback records."""

import sqlalchemy as sa

from alembic import op

revision = "0003_feedback"
down_revision = "0002_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_feedback",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("answer_id", sa.String(length=36), nullable=False),
        sa.Column("rating", sa.String(length=20), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_chat_feedback_answer_id", "chat_feedback", ["answer_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_feedback_answer_id", table_name="chat_feedback")
    op.drop_table("chat_feedback")
