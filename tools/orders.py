
"""Database access and safe, runtime-discovered order schema metadata."""

from __future__ import annotations

import re
import sqlite3

from config import Settings, get_settings


_SENSITIVE_NAME_PARTS = (
    "customer", "email", "phone", "mobile", "address", "payment",
    "card", "cvv", "bank", "account", "password", "secret",
    "token", "credential", "auth", "session", "cookie", "ssn",
    "social_security", "birth", "dob", "ip_address", "internal",
    "private", "metadata", "debug", "billing", "shipping", "ship_to",
    "recipient", "contact", "first_name", "last_name", "full_name",
    "notes", "comment", "message", "description",
)


class OrderStoreError(RuntimeError):
    """Raised when the order database or its schema cannot be used."""


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _is_safe_order_column(name: str, declared_type: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")

    return bool(normalized) and normalized != "customer_id" and not any(
        part in normalized for part in _SENSITIVE_NAME_PARTS
    ) and "BLOB" not in declared_type.upper()


def _semantic_type(name: str, declared_type: str) -> str:
    """Map SQLite declarations and common field names to resolver types."""
    name_parts = name.casefold().split("_")
    sql_type = (declared_type or "TEXT").upper()

    if any(part in {"date", "time", "timestamp"} for part in name_parts):
        return "date"
    if (
        sql_type == "BOOLEAN"
        or name_parts[0] in {"is", "has"}
        or name_parts[-1:] in (["enabled"], ["active"])
    ):
        return "boolean"
    if any(
        part in {
            "price", "amount", "quantity", "qty", "count", "total",
            "cost", "weight", "rating", "score",
        }
        for part in name_parts
    ):
        return "number"
    if "INT" in sql_type:
        return "integer"
    if any(token in sql_type for token in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "number"
    if any(token in sql_type for token in ("DATE", "TIME")):
        return "date"
    return "text"


def _connect(settings: Settings):
    db_path = settings.orders_db_path

    if not db_path.is_file():
        raise OrderStoreError("Order database is unavailable.")

    try:
        connection = sqlite3.connect(
            f"{db_path.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection
    except sqlite3.Error as exc:
        raise OrderStoreError("Could not open the order database.") from exc


def _columns(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = connection.execute(
            f"PRAGMA table_info({_quote_identifier('orders')})"
        ).fetchall()
    except sqlite3.Error as exc:
        raise OrderStoreError("Could not inspect the order database schema.") from exc

    if not rows:
        raise OrderStoreError("The orders table is missing or has no columns.")

    return {
        str(row["name"]): str(row["type"] or "TEXT")
        for row in rows
    }


def order_schema(settings: Settings | None = None) -> list[dict[str, str]]:
    """Return safe field names and normalized types without reading order values."""
    config = settings or get_settings()

    with _connect(config) as connection:
        columns = _columns(connection)

    return [
        {"name": name, "type": _semantic_type(name, declared_type)}
        for name, declared_type in columns.items()
        if _is_safe_order_column(name, declared_type)
    ]


def customer_orders(
    user_id: str,
    settings: Settings | None = None,
) -> list[dict]:
    """Return safe orders belonging only to the authenticated customer."""
    if not user_id:
        raise OrderStoreError("An authenticated customer ID is required.")

    config = settings or get_settings()

    with _connect(config) as connection:
        columns = _columns(connection)

        if "customer_id" not in columns or "order_id" not in columns:
            raise OrderStoreError("The orders table is missing required columns.")

        fields = [
            name
            for name, declared_type in columns.items()
            if _is_safe_order_column(name, declared_type)
        ]

        if "order_id" not in fields:
            raise OrderStoreError("The orders table has no safe order ID field.")

        selected_fields = ", ".join(map(_quote_identifier, fields))
        order_clause = (
            f"{_quote_identifier('order_date')} DESC, "
            f"{_quote_identifier('order_id')}"
            if "order_date" in fields
            else _quote_identifier("order_id")
        )

        try:
            rows = connection.execute(
                f"""
                SELECT {selected_fields}
                FROM {_quote_identifier('orders')}
                WHERE {_quote_identifier('customer_id')} = ?
                  AND {_quote_identifier('order_id')} IS NOT NULL
                  AND TRIM({_quote_identifier('order_id')}) != ''
                ORDER BY {order_clause}
                """,
                (user_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise OrderStoreError("Could not retrieve customer orders.") from exc

    return [dict(row) for row in rows]


def check_order(
    order_id: str,
    user_id: str,
    settings: Settings | None = None,
) -> dict:
    """Validate an order against the authenticated customer's orders."""
    if not order_id or not user_id:
        return {"found": False}

    requested_id = str(order_id).strip().upper().replace(" ", "")

    for order in customer_orders(user_id, settings):
        existing_id = str(order.get("order_id") or "").strip().upper().replace(" ", "")
        if existing_id == requested_id:
            return {"found": True, "order_id": order["order_id"]}

    return {"found": False}
