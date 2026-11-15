
"""Structured schemas for the Customer Support Resolution Copilot."""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


# Configurable field-specific matching and normalization behaviour.
FILTER_FIELD_METADATA = {
    "product": {"match": "partial"},
    "variant": {"match": "contains"},
    "status": {"aliases": {"delievered": "delivered"}},
}


ExpressionOperator = Literal[
    "AND", "OR", "NOT",
    "eq", "ne", "in", "not_in",
    "contains", "not_contains",
    "gt", "gte", "lt", "lte",
    "between", "not_between",
    "before", "after",
    "is_null", "is_not_null",
]


class FilterExpression(BaseModel):
    """A comparison leaf or a recursively validated logical group."""

    operator: ExpressionOperator = Field(
        description=(
            "AND, OR, or NOT for logical groups; "
            "otherwise a comparison operator."
        )
    )
    field: str | None = Field(
        default=None,
        description="Database field name for a comparison condition.",
    )
    value: Any = Field(
        default=None,
        description="Comparison value. Use null for null checks.",
    )
    conditions: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "One or more complete expressions for AND/OR; "
            "exactly one for NOT."
        ),
    )

    @model_validator(mode="after")
    def validate_expression(self):
        if self.operator in {"AND", "OR", "NOT"}:
            if self.field is not None or self.value is not None:
                raise ValueError(
                    "Logical groups cannot contain field or value."
                )

            if not self.conditions:
                raise ValueError(
                    f"{self.operator} requires non-empty conditions."
                )

            if self.operator == "NOT" and len(self.conditions) != 1:
                raise ValueError("NOT requires exactly one condition.")

            validated_conditions = []

            for condition in self.conditions:
                if not isinstance(condition, dict):
                    raise ValueError(
                        "Every logical condition must be an object."
                    )

                validated = FilterExpression.model_validate(condition)
                validated_conditions.append(validated.model_dump())

            self.conditions = validated_conditions

        else:
            if not self.field or not self.field.strip():
                raise ValueError(
                    "Comparison conditions require a field."
                )

            if self.conditions is not None:
                raise ValueError(
                    "Comparison conditions cannot contain nested conditions."
                )

        return self


class OrderReference(BaseModel):
    """Structured interpretation of an order-related request."""

    order_id: str | None = None
    product: str | None = None
    variant: str | None = None
    status: str | None = None

    date_field: str | None = Field(
        default=None,
        description=(
            "Exact schema field representing the relevant date. "
            "Select it from the supplied order schema when needed."
        ),
    )
    date_from: str | None = None
    date_to: str | None = None

    position_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based position in the sorted matching orders. "
            "First is 0, second is 1, third is 2, and so on. "
            "Use this for requests such as latest, oldest, "
            "second latest, or third oldest."
        ),
    )

    sort_by: str | None = Field(
        description=(
            "Required output field. When the customer requests sorting, "
            "populate this with the exact matching field name from the "
            "supplied database schema. Use null only when sorting is "
            "not requested."
        ),
    )
    sort_order: Literal["asc", "desc"] | None = Field(
        description=(
            "Required output field. When sorting is requested, populate "
            "this together with sort_by. Use asc for lowest to highest, "
            "smallest to largest, oldest to newest, or A to Z. Use desc "
            "for highest to lowest, largest to smallest, newest to oldest, "
            "or Z to A. Use null only when sorting is not requested."
        ),
    )

    filters: FilterExpression | None = None
    requested_fields: list[str] = Field(default_factory=list)

    wants_all: bool = False
    list_orders: bool = False
    ambiguous: bool = False
    needs_clarification: bool = False
    clarification_choice: int | None = Field(
        default=None,
        ge=1,
        le=4,
    )

    @field_validator("sort_by", "date_field")
    @classmethod
    def validate_schema_field(cls, value):
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Schema field names cannot be empty.")
        return value

    @field_validator("filters", mode="before")
    @classmethod
    def normalize_legacy_filters(cls, value):
        """Convert legacy field-based filters into expression trees."""

        if value is None or value == {}:
            return None

        if isinstance(value, FilterExpression):
            return value

        if not isinstance(value, dict):
            return value

        if "operator" in value or "field" in value:
            return value

        leaves = []

        for field, specification in value.items():
            if not isinstance(field, str) or not field.strip():
                return value

            if (
                isinstance(specification, list)
                and specification
                and all(
                    isinstance(item, dict) and "operator" in item
                    for item in specification
                )
            ):
                for condition in specification:
                    leaves.append({
                        "field": field,
                        "operator": condition["operator"],
                        "value": condition.get("value"),
                    })
                continue

            if isinstance(specification, list):
                leaves.append({
                    "field": field,
                    "operator": "in",
                    "value": specification,
                })
                continue

            if (
                isinstance(specification, dict)
                and "operator" in specification
            ):
                leaves.append({
                    "field": field,
                    "operator": specification["operator"],
                    "value": specification.get("value"),
                })
                continue

            leaves.append({
                "field": field,
                "operator": "eq",
                "value": (
                    specification.get("value")
                    if isinstance(specification, dict)
                    and "value" in specification
                    else specification
                ),
            })

        if not leaves:
            return None

        if len(leaves) == 1:
            return leaves[0]

        return {
            "operator": "AND",
            "conditions": leaves,
        }


class GatekeeperResult(BaseModel):
    """Structured safety, routing, and evidence-requirement decision."""

    safety: str = Field(pattern="^(SAFE|UNSAFE)$")
    scope: str = Field(
        pattern="^(none|out_of_scope|order_status_lookup|in_scope)$"
    )
    intent: str | None = None
    references_current_order: bool = False
    requires_policy: bool = False
    order_reference: OrderReference = Field(
        default_factory=OrderReference
    )


class ExecutionEvent(TypedDict, total=False):
    node: str
    status: str
    summary: str
    latency_seconds: float
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    error_type: str
    error_status: str
    provider_error_code: str
    error_message: str
    model_calls: list[dict[str, Any]]


class ConversationMessage(TypedDict):
    role: str
    content: str


class CopilotState(TypedDict, total=False):
    user_message: str
    conversation_history: list[ConversationMessage]

    safety: str
    scope: str | None
    intent: str | None
    references_current_order: bool
    requires_policy: bool
    order_reference: dict[str, Any]
    gatekeeper_status: str

    order_info: list[dict[str, Any]]
    order_lookup_status: str
    current_order: str | None
    clarification_options: list[dict[str, Any]]
    recent_order_options: list[dict[str, Any]]
    selected_order_id: str | None
    clarification_original_question: str

    user_id: str | None
    auth_state: str
    evidence: list[dict[str, Any]]
    retrieval_status: str

    response: str
    response_status: str
    workflow_status: str
    execution: list[ExecutionEvent]
    errors: list[str]
