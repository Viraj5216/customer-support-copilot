"""Idempotently migrate synthetic order ownership and seed demo customers."""
from __future__ import annotations

import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from argon2 import PasswordHasher

DEMO_USERS = (
    ("demo-customer-a", "Demo Customer A", "demo-a@example.test"),
    ("demo-customer-b", "Demo Customer B", "demo-b@example.test"),
    ("demo-customer-single", "Demo Customer Single", "demo-single@example.test"),
    ("demo-customer-open", "Demo Customer Open", "demo-open@example.test"),
)
PRODUCTS = (
    ("AeroBrew X2 Coffee Maker", "Black / 12 cup", 1, 89.99),
    ("AeroBrew X2 Coffee Maker", "Silver / 10 cup", 2, 94.50),
    ("Northstar Trail Pack 28L", "Blue / Medium", 1, 64.00),
    ("LumaDesk LED Lamp", "White / Standard", 1, 39.95),
    ("SonicWave Headphones Pro", "Black / Large", 1, 129.00),
    ("TerraSteel Water Bottle", "Green / 750 ml", 2, 32.50),
)


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _migrate_orders(connection: sqlite3.Connection, user_ids: list[str]) -> bool:
    columns = _table_columns(connection, "orders")
    if not columns:
        raise RuntimeError("Orders table is unavailable")
    foreign_keys = connection.execute("PRAGMA foreign_key_list(orders)").fetchall()
    already_migrated = (
        "customer_id" in columns
        and any(row[2] == "users" and row[3] == "customer_id" for row in foreign_keys)
        and "customer_email" not in columns
        and {"order_date", "delivery_date", "quantity", "price", "variant"}.issubset(columns)
    )
    if already_migrated:
        return False

    selected = [name if name in columns else f"NULL AS {name}" for name in (
        "order_id", "status", "issue", "product", "customer_id", "order_date", "delivery_date", "quantity", "price", "variant"
    )]
    rows = connection.execute(
        "SELECT " + ", ".join(selected) + " FROM orders ORDER BY order_id"
    ).fetchall()
    connection.execute("DROP TABLE IF EXISTS orders_new")
    connection.execute(
        "CREATE TABLE orders_new ("
        "order_id TEXT PRIMARY KEY, status TEXT, issue TEXT, product TEXT, "
        "customer_id TEXT REFERENCES users(user_id), order_date TEXT, delivery_date TEXT, "
        "quantity INTEGER, price REAL, variant TEXT)"
    )
    assigned = {row[4] for row in rows if row[4] in user_ids}
    if len({row[0] for row in rows if row[0] is not None and str(row[0]).strip()}) >= 2:
        for user_id in user_ids:
            if user_id not in assigned:
                replace_index = next(
                    (index for index, row in enumerate(rows)
                     if row[0] is not None and str(row[0]).strip() and row[4] in (None, "", *user_ids)),
                    None,
                )
                if replace_index is not None:
                    old = rows[replace_index]
                    rows[replace_index] = (*old[:4], user_id, *old[5:])
                    assigned.add(user_id)
    index = 0
    for order_id, status, issue, product, owner, order_date, delivery_date, quantity, price, variant in rows:
        if owner not in user_ids and order_id is not None and str(order_id).strip():
            owner = user_ids[index % len(user_ids)]
            index += 1
        if order_id is None or not str(order_id).strip():
            owner = None
        connection.execute(
            "INSERT INTO orders_new(order_id,status,issue,product,customer_id,order_date,delivery_date,quantity,price,variant) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (order_id, status, issue, product, owner, order_date, delivery_date, quantity, price, variant),
        )
    connection.execute("DROP TABLE orders")
    connection.execute("ALTER TABLE orders_new RENAME TO orders")
    return True


def _seed_order_rows(connection: sqlite3.Connection, now: datetime) -> None:
    """Fill missing synthetic records idempotently; dates are relative to injected seed time."""
    today = now.date()
    # The named fixture rows guarantee duplicate and edge-case scenarios even when
    # migrating a database that already contains a few legacy synthetic orders.
    seeds = [
        (DEMO_USERS[0][0], "7001", "AeroBrew X2 Coffee Maker", "Delivered", 30, 3, 1, 89.99, "Black / 12 cup"),
        (DEMO_USERS[0][0], "7002", "AeroBrew X2 Coffee Maker", "Shipped", 2, None, 2, 94.50, "Silver / 10 cup"),
        (DEMO_USERS[0][0], "7003", "Northstar Trail Pack 28L", "Delivered", 18, 3, 1, 64.00, "Blue / Medium"),
        (DEMO_USERS[0][0], "7004", "LumaDesk LED Lamp", "Processing", 9, None, 1, 39.95, "White / Standard"),
        (DEMO_USERS[0][0], "7005", "SonicWave Headphones Pro", "Delivered", 6, 3, 1, 129.00, "Black / Large"),
        (DEMO_USERS[0][0], "7006", "TerraSteel Water Bottle", "Delivered", 4, 3, 2, 32.50, "Green / 750 ml"),
        (DEMO_USERS[1][0], "8001", "LumaDesk LED Lamp", "Delivered", 5, 3, 1, 39.95, "Black / Standard"),
        (DEMO_USERS[1][0], "8002", "LumaDesk LED Lamp", "Processing", 5, None, 2, 41.95, "Brass / Standard"),
        (DEMO_USERS[1][0], "8003", "AeroBrew X2 Coffee Maker", "Shipped", 12, None, 1, 89.99, "Black / 12 cup"),
        (DEMO_USERS[1][0], "8004", "SonicWave Headphones Pro", "Delivered", 21, 3, 1, 129.00, "Black / Large"),
        (DEMO_USERS[1][0], "8005", "Northstar Trail Pack 28L", "Delivered", 40, 3, 1, 64.00, "Blue / Medium"),
        (DEMO_USERS[2][0], "9001", "Solo Desk Fan", "Delivered", 12, 3, 1, 20.00, "White / Standard"),
        (DEMO_USERS[3][0], "9002", "Open Parcel", "Shipped", 2, None, 1, 50.00, "Standard"),
    ]
    for user_id, suffix, product, status, days, delivery_days, quantity, price, variant in seeds:
        order_date = today - timedelta(days=days)
        delivery_date = (order_date + timedelta(days=delivery_days)).isoformat() if delivery_days is not None else None
        connection.execute(
            "INSERT OR IGNORE INTO orders(order_id,status,issue,product,customer_id,order_date,delivery_date,quantity,price,variant) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (f"ORD{suffix}", status, "", product, user_id, order_date.isoformat(), delivery_date, quantity, price, variant),
        )
    # Enrich any migrated legacy rows without overwriting fields already present.
    for owner_index, (user_id, _name, _email) in enumerate(DEMO_USERS[:2]):
        existing = connection.execute(
            "SELECT order_id FROM orders WHERE customer_id=? AND order_id NOT LIKE 'ORDDEMO%' ORDER BY order_id", (user_id,)
        ).fetchall()
        for index, (order_id,) in enumerate(existing):
            product, variant, quantity, price = PRODUCTS[index % len(PRODUCTS)]
            order_date = today - timedelta(days=45 + owner_index * 2 + index)
            delivered = index % 2 == 0
            connection.execute(
                "UPDATE orders SET order_date=COALESCE(order_date,?), delivery_date=COALESCE(delivery_date,?), "
                "quantity=COALESCE(quantity,?), price=COALESCE(price,?), variant=COALESCE(variant,?), "
                "product=COALESCE(NULLIF(product,''),?), status=COALESCE(NULLIF(status,''),?) WHERE order_id=? "
                "AND (order_date IS NULL OR quantity IS NULL OR price IS NULL OR variant IS NULL "
                "OR product IS NULL OR product='' OR status IS NULL OR status='')",
                (order_date.isoformat(), (order_date + timedelta(days=3)).isoformat() if delivered else None,
                 quantity, price, variant, product, "Delivered" if delivered else "Processing", order_id),
            )


def seed_demo_users(database_path: Path, credential_path: Path, *, now: datetime | None = None) -> bool:
    """Create the schema/users and assign orders; write generated passwords once.

    Re-running after the local credential file has been created performs only
    idempotent schema/ownership checks and never changes existing password hashes.
    """
    database_path = Path(database_path)
    credential_path = Path(credential_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    credentials_exist = credential_path.exists()
    existing_credentials = {}
    if credentials_exist:
        existing_credentials = {
            line.split("\t", 1)[0]: line.rstrip("\n").split("\t", 1)[1]
            for line in credential_path.read_text(encoding="utf-8").splitlines()
            if "\t" in line
        }
    credentials = {
        email: existing_credentials.get(email, secrets.token_urlsafe(24))
        for _, _, email in DEMO_USERS
    }
    hasher = PasswordHasher()
    migrated = False
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, "
            "login_email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS orders ("
            "order_id TEXT PRIMARY KEY, status TEXT, issue TEXT, product TEXT, "
            "customer_id TEXT REFERENCES users(user_id), order_date TEXT, "
            "delivery_date TEXT, quantity INTEGER, price REAL, variant TEXT)"
        )
        existing_users = {
            row[0]: row[1]
            for row in connection.execute("SELECT user_id, login_email FROM users")
        }
        if credentials_exist and any(
            user_id in existing_users and existing_users[user_id] not in existing_credentials
            for user_id, _name, _email in DEMO_USERS
        ):
            raise RuntimeError("An existing demo account has no entry in the local credentials file")
        if not credentials_exist and any(user_id in existing_users for user_id, _, _ in DEMO_USERS):
            raise RuntimeError("Demo accounts already exist without the local credential file")
        for user_id, display_name, email in DEMO_USERS:
            if user_id not in existing_users:
                connection.execute(
                    "INSERT INTO users(user_id,display_name,login_email,password_hash) VALUES(?,?,?,?)",
                    (user_id, display_name, email, hasher.hash(credentials[email])),
                )
        user_ids = [item[0] for item in DEMO_USERS]
        # Legacy unowned sample records stay with the regular multi-order demo accounts.
        migrated = _migrate_orders(connection, user_ids[:2])
        # A newly created users table must not coexist with unassigned real rows.
        rows = connection.execute(
            "SELECT order_id FROM orders WHERE order_id IS NOT NULL "
            "AND TRIM(order_id) != '' AND customer_id IS NULL ORDER BY order_id"
        ).fetchall()
        for index, (order_id,) in enumerate(rows):
            connection.execute(
                "UPDATE orders SET customer_id = ? WHERE order_id = ?",
                (user_ids[index % 2], order_id),
            )
        seed_now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
        if seed_now.tzinfo is None:
            raise ValueError("Seed clock must return a timezone-aware datetime")
        _seed_order_rows(connection, seed_now)
        connection.commit()

    missing_credential_users = [
        (user_id, display_name, email) for user_id, display_name, email in DEMO_USERS
        if email not in existing_credentials
    ]
    if missing_credential_users:
        temp_path = credential_path.with_suffix(credential_path.suffix + ".tmp")
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                if credentials_exist:
                    handle.write(credential_path.read_text(encoding="utf-8"))
                for _user_id, _display_name, email in missing_credential_users:
                    handle.write(f"{email}\t{credentials[email]}\n")
            os.replace(temp_path, credential_path)
            os.chmod(credential_path, 0o600)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    return migrated


def main() -> None:
    from config import REPO_ROOT, get_settings

    config = get_settings()
    credentials = REPO_ROOT / "data/demo_credentials.txt"
    changed = seed_demo_users(config.orders_db_path, credentials)
    print("Synthetic demo accounts ready; credentials saved in the ignored local file.")
    if changed:
        print("Customer ownership schema migrated.")


if __name__ == "__main__":
    main()
