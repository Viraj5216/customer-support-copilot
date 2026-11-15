"""Idempotent schema migration and synthetic demo-user seed entry point."""
from pathlib import Path

from scripts.seed_demo_users import seed_demo_users


def migrate_and_seed(database_path: Path, credential_path: Path) -> bool:
    return seed_demo_users(database_path, credential_path)


def main() -> None:
    from config import REPO_ROOT, get_settings
    config = get_settings()
    migrate_and_seed(config.orders_db_path, REPO_ROOT / "data/demo_credentials.txt")
    print("Synthetic customer ownership migration complete.")


if __name__ == "__main__":
    main()
