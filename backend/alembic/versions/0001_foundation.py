"""Initial database revision; application tables are introduced by later phases."""

revision = "0001_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Establishes a versioned migration baseline without premature business schema.
    pass


def downgrade() -> None:
    pass
