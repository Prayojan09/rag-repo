"""Align Google identity uniqueness with the ORM's unique indexes."""

import sqlalchemy as sa

from alembic import op

revision = "0011_auth_unique_indexes"
down_revision = "0010_google_auth_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Replace duplicate unique constraints and ordinary indexes with unique indexes."""
    inspector = sa.inspect(op.get_bind())
    unique_columns = {"google_sub", "tenant_id"}
    for constraint in inspector.get_unique_constraints("auth_users"):
        if set(constraint.get("column_names", ())) in ({"google_sub"}, {"tenant_id"}):
            op.drop_constraint(constraint["name"], "auth_users", type_="unique")
    for index in inspector.get_indexes("auth_users"):
        if index["name"] in {"ix_auth_users_google_sub", "ix_auth_users_tenant_id"}:
            op.drop_index(index["name"], table_name="auth_users")
    for column in sorted(unique_columns):
        op.create_index(f"ix_auth_users_{column}", "auth_users", [column], unique=True)


def downgrade() -> None:
    """Restore the unique constraints and ordinary lookup indexes from revision 0010."""
    for column in ("google_sub", "tenant_id"):
        op.drop_index(f"ix_auth_users_{column}", table_name="auth_users")
        op.create_unique_constraint(f"auth_users_{column}_key", "auth_users", [column])
        op.create_index(f"ix_auth_users_{column}", "auth_users", [column])
