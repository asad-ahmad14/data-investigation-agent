"""Load the deterministic synthetic retail dataset (replaces any existing retail data).

Usage:  python scripts/seed_db.py
Needs:  migrations applied first (alembic upgrade head). Uses the admin role from .env.
"""

from app.core.config import get_settings
from app.db.engines import get_admin_engine
from app.retail.seed import seed_database


def main() -> None:
    get_settings()  # fail fast, with a clear message, on bad configuration
    counts = seed_database(get_admin_engine())
    for table, n in counts.items():
        print(f"{table:12s} {n:>7,d} rows")
    print("Seed complete (synthetic data).")


if __name__ == "__main__":
    main()
