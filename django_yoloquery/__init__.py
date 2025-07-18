"""Django YOLOQuery - LLM-backed natural language queries for Django ORM."""

from __future__ import annotations

import json
import os
import logging
import datetime as dt
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Set,
    Type,
    TypeVar,
    cast,
)

from django.db import models
from django.db.models import Q
from django.apps import apps as django_apps

try:
    from django.conf import settings as dj_settings
except Exception:
    dj_settings = None

__version__ = "0.1.0"
M = TypeVar("M", bound=models.Model)
FilterValue = Any
LookupName = str
FieldPath = str
Patterns = Sequence[str]
SchemaDict = Dict[str, Any]

logger = logging.getLogger(__name__)

_DEFAULT_SCHEMA_DEPTH = 1
_DEFAULT_INCLUDE_REVERSE = True
_SETTINGS_DEPTH_NAME = "YOLOQUERY_SCHEMA_DEPTH"
_SETTINGS_INCLUDE_NAME = "YOLOQUERY_INCLUDE_REVERSE"
_SETTINGS_AUTOINSTALL_NAME = "YOLOQUERY_AUTO_INSTALL"
_SETTINGS_MODEL_NAME = "YOLOQUERY_LLM_MODEL"
_SETTINGS_OPENAI_KEY = "YOLOQUERY_OPENAI_API_KEY"


def _get_env_bool(name: str) -> Optional[bool]:
    env = os.getenv(name)
    if env is None:
        return None
    return env.strip().lower() in {"1", "true", "t", "yes", "y"}


def get_default_schema_depth() -> int:
    env = os.getenv(_SETTINGS_DEPTH_NAME)
    try_env = int(env) if env is not None else None
    try:
        if dj_settings is not None and hasattr(dj_settings, _SETTINGS_DEPTH_NAME):
            return int(getattr(dj_settings, _SETTINGS_DEPTH_NAME))
    except Exception:  # nosec B110 - intentionally catching all exceptions for config fallback
        pass
    if try_env is not None:
        return try_env
    return _DEFAULT_SCHEMA_DEPTH


def get_default_include_reverse() -> bool:
    try:
        if dj_settings is not None and hasattr(dj_settings, _SETTINGS_INCLUDE_NAME):
            return bool(getattr(dj_settings, _SETTINGS_INCLUDE_NAME))
    except Exception:  # nosec B110 - intentionally catching all exceptions for config fallback
        pass
    env_val = _get_env_bool(_SETTINGS_INCLUDE_NAME)
    if env_val is not None:
        return env_val
    return _DEFAULT_INCLUDE_REVERSE


def get_auto_install_patterns() -> Patterns:
    try:
        if dj_settings is not None and hasattr(dj_settings, _SETTINGS_AUTOINSTALL_NAME):
            pats = getattr(dj_settings, _SETTINGS_AUTOINSTALL_NAME)
            if isinstance(pats, (list, tuple)):
                return list(pats)
    except Exception:  # nosec B110 - intentionally catching all exceptions for config fallback
        pass
    env_val = os.getenv(_SETTINGS_AUTOINSTALL_NAME)
    if env_val:
        return [p.strip() for p in env_val.split(",") if p.strip()]
    return []


def get_openai_model_name() -> str:
    try:
        if dj_settings is not None and hasattr(dj_settings, _SETTINGS_MODEL_NAME):
            return str(getattr(dj_settings, _SETTINGS_MODEL_NAME))
    except Exception:  # nosec B110 - intentionally catching all exceptions for config fallback
        pass
    return os.getenv(_SETTINGS_MODEL_NAME) or "gpt-4o-mini"


def get_openai_api_key() -> Optional[str]:
    try:
        if dj_settings is not None and hasattr(dj_settings, _SETTINGS_OPENAI_KEY):
            key = getattr(dj_settings, _SETTINGS_OPENAI_KEY)
            if key:
                return str(key)
    except Exception:  # nosec B110 - intentionally catching all exceptions for config fallback
        pass
    key = os.getenv(_SETTINGS_OPENAI_KEY) or os.getenv("OPENAI_API_KEY")
    return key


class AIQueryError(Exception):
    """Base exception for AI Query issues."""


class AIQueryModelMismatchError(AIQueryError):
    """LLM returned a field not present / not allowed on the model."""

    def __init__(self, field_name: str):
        self.field_name = field_name
        super().__init__(f"Unknown or disallowed field: {field_name}.")


class AIQueryOperatorError(AIQueryError):
    def __init__(self, field_name: str, op: str):
        self.field_name = field_name
        self.op = op
        super().__init__(f"Operator '{op}' not allowed for field '{field_name}'.")


class AIQueryValueError(AIQueryError):
    def __init__(self, field_name: str, value: Any, msg: str = "Invalid value"):
        self.field_name = field_name
        self.value = value
        super().__init__(f"{msg} for field '{field_name}': {value!r}")


class AIQueryLLMError(AIQueryError):
    """Raised when the LLM signals failure or returns bad JSON."""

    pass


@dataclass
class FilterSpec:
    path: FieldPath
    op: LookupName = "iexact"
    value: FilterValue = None


@dataclass
class IntentSpec:
    status: str = "ok"
    message: str = ""
    logic: str = "and"
    filters: List[FilterSpec] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    limit: Optional[int] = None

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "IntentSpec":
        status = (data.get("status") or "ok").lower()
        message = cast(str, data.get("message") or "")
        logic = (data.get("logic") or "and").lower()
        filters_raw = data.get("filters") or []
        filters: List[FilterSpec] = []
        if isinstance(filters_raw, Sequence):
            for c in filters_raw:
                try:
                    filters.append(FilterSpec(path=c["path"], op=c.get("op", "iexact"), value=c.get("value")))
                except Exception:  # nosec B112 - intentionally catching all exceptions to skip invalid filters
                    continue
        order = list(data.get("order_by") or [])
        limit = data.get("limit")
        return cls(status=status, message=message, logic=logic, filters=filters, order_by=order, limit=limit)


DEFAULT_TYPE_LOOKUPS: Mapping[str, Set[str]] = {
    "CharField": {"exact", "iexact", "icontains", "istartswith", "iendswith", "in", "isnull", "ne"},
    "TextField": {"icontains", "iexact", "in", "isnull", "ne"},
    "EmailField": {"iexact", "icontains", "in", "isnull", "ne"},
    "SlugField": {"iexact", "icontains", "in", "isnull", "ne"},
    "UUIDField": {"exact", "in", "isnull", "ne"},
    "IntegerField": {"exact", "gt", "gte", "lt", "lte", "in", "isnull", "ne"},
    "BigIntegerField": {"exact", "gt", "gte", "lt", "lte", "in", "isnull", "ne"},
    "AutoField": {"exact", "in", "ne"},
    "BooleanField": {"exact", "isnull", "ne"},
    "DateField": {"exact", "gt", "gte", "lt", "lte", "range", "isnull", "ne"},
    "DateTimeField": {"exact", "gt", "gte", "lt", "lte", "range", "isnull", "ne"},
    "DecimalField": {"exact", "gt", "gte", "lt", "lte", "in", "isnull", "ne"},
    "FloatField": {"exact", "gt", "gte", "lt", "lte", "in", "isnull", "ne"},
    "ForeignKey": {"exact", "in", "isnull", "ne"},
    "OneToOneField": {"exact", "isnull", "ne"},
    "ManyToManyField": {"in", "isnull", "ne"},
}

OP_ALIAS: Mapping[str, str] = {
    "eq": "exact",
    "==": "exact",
    "=": "exact",
    "ne": "ne",
    "!=": "ne",
    "not": "ne",
    "contains": "icontains",
    "startswith": "istartswith",
    "endswith": "iendswith",
    ">": "gt",
    ">=": "gte",
    "after": "gt",
    "since": "gte",
    "<": "lt",
    "<=": "lte",
    "before": "lt",
    "null": "isnull",
    "none": "isnull",
    "bool": "exact",
}


def _field_category(field: models.Field) -> str:
    return field.__class__.__name__


REL_FIELD_TYPES = (models.ForeignKey, models.OneToOneField, models.ManyToManyField)


def build_schema_for_model(
    model: Type[models.Model],
    *,
    depth: int = _DEFAULT_SCHEMA_DEPTH,
    include_reverse: bool = _DEFAULT_INCLUDE_REVERSE,
    synonyms: Optional[Dict[str, Sequence[str]]] = None,
) -> SchemaDict:
    """Return nested JSON schema for model + related models."""
    seen: Dict[Type[models.Model], Dict[str, Any]] = {}

    def _syns(m: Type[models.Model], fname: str) -> List[str]:
        if not synonyms:
            return []
        key = f"{m.__name__}.{fname}"
        if key in synonyms:
            return list(synonyms[key])
        return list(synonyms.get(fname, []))

    def _walk(m: Type[models.Model], d: int) -> Dict[str, Any]:
        if m in seen:
            return seen[m]
        info: Dict[str, Any] = {"fields": {}, "relationships": {}}
        seen[m] = info
        for f in m._meta.get_fields(include_parents=True, include_hidden=False):
            if getattr(f, "auto_created", False) and not getattr(f, "concrete", True):
                if include_reverse:
                    accessor = f.get_accessor_name()
                    rel_model = f.related_model
                    info["relationships"][accessor] = {"type": "REV", "to": rel_model.__name__}
                    if d > 0:
                        _walk(rel_model, d - 1)
                continue

            if not getattr(f, "is_relation", False):
                info["fields"][f.name] = {
                    "type": _field_category(f),
                    "null": getattr(f, "null", False),
                    "synonyms": _syns(m, f.name),
                }
                if hasattr(f, "max_length") and getattr(f, "max_length"):
                    info["fields"][f.name]["max_length"] = getattr(f, "max_length")
                continue

            if d <= 0:
                continue

            rel_model = f.related_model
            if isinstance(f, models.ForeignKey):
                info["relationships"][f.name] = {"type": "FK", "to": rel_model.__name__}
            elif isinstance(f, models.OneToOneField):
                info["relationships"][f.name] = {"type": "O2O", "to": rel_model.__name__}
            elif isinstance(f, models.ManyToManyField):
                info["relationships"][f.name] = {"type": "M2M", "to": rel_model.__name__}
            else:
                info["relationships"][f.name] = {"type": "REL", "to": rel_model.__name__}
            _walk(rel_model, d - 1)
        return info

    _walk(model, depth)
    return {
        "root_model": model.__name__,
        "app_label": model._meta.app_label,
        "models": {m.__name__: seen[m] for m in seen},
    }


def _coerce_value(field: models.Field, value: Any) -> Any:
    if value is None:
        return None
    cat = _field_category(field)
    if cat in {"CharField", "TextField", "EmailField", "SlugField", "UUIDField"}:
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return str(value)
    if cat in {"IntegerField", "BigIntegerField", "AutoField"}:
        if isinstance(value, (list, tuple)):
            try:
                return [int(v) for v in value]
            except Exception as exc:
                raise AIQueryValueError(field.name, value) from exc
        try:
            return int(value)
        except Exception as exc:
            raise AIQueryValueError(field.name, value) from exc
    if cat in {"BooleanField"}:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "t", "1", "yes", "y"}:
                return True
            if v in {"false", "f", "0", "no", "n"}:
                return False
        raise AIQueryValueError(field.name, value)
    if cat in {"DateField", "DateTimeField"}:
        if isinstance(value, str):
            try:
                if cat == "DateField":
                    return dt.date.fromisoformat(value)
                else:
                    return dt.datetime.fromisoformat(value)
            except ValueError as exc:
                raise AIQueryValueError(field.name, value, msg="Bad ISO date") from exc
        if isinstance(value, (int, float)):
            if cat == "DateField":
                return dt.date.fromtimestamp(value)
            else:
                return dt.datetime.fromtimestamp(value)
        raise AIQueryValueError(field.name, value)
    if cat in {"ForeignKey", "OneToOneField"}:
        pk_field = field.target_field
        return _coerce_value(pk_field, value)
    if cat in {"ManyToManyField"}:
        if not isinstance(value, (list, tuple)):
            value = [value]
        pk_field = field.target_field
        return [_coerce_value(pk_field, v) for v in value]
    return value


class LLMClient:
    """Translate natural language to IntentSpec using OpenAI LLM."""

    def __init__(self, model_name: str | None = None, client: Any | None = None, *, temperature: float = 0.0) -> None:
        self.model_name = model_name or get_openai_model_name()
        self._client: Any | None = client
        self.temperature = temperature

    _SYSTEM_TEMPLATE = (
        "You convert natural-language data queries into strict JSON filter specs for a Django ORM query. "
        "RULES: Respond with JSON *only* (no markdown). Use only the fields/relationships in the supplied SCHEMA. "
        "Use Django double-underscore paths to traverse relations. Default to case-insensitive matching for text. "
        "Supported ops: exact, iexact, icontains, istartswith, iendswith, gt, gte, lt, lte, in, range, isnull, ne. "
        "Dates must be YYYY-MM-DD. 'in' expects JSON array. 'range' expects [start,end]. 'isnull' expects true/false. "
        'IF YOU CANNOT PRODUCE A VALID FILTER (missing info, ambiguous, nonsense), '
        'return {"status":"error", "message":"..."} and nothing else. '
        "Return raw JSON now.")

    _USER_TEMPLATE = "SCHEMA:\n{schema}\n\nUSER QUERY:\n{query}\n"
    _INTENT_JSON_SCHEMA: Dict[str, Any] = {
        "name": "YoloIntent",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["ok", "error"]},
                "message": {"type": "string", "default": ""},
                "logic": {"type": "string", "enum": ["and", "or"], "default": "and"},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"path": {"type": "string"}, "op": {"type": "string"}, "value": {}},
                        "required": ["path"],
                    },
                    "default": [],
                },
                "order_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "limit": {"type": ["integer", "null"], "minimum": 1},
            },
            "required": ["status"],
        },
    }

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("openai package not installed; install openai >=1.0") from exc
        api_key = get_openai_api_key()
        if not api_key:
            raise RuntimeError("OpenAI API key not configured; set YOLOQUERY_OPENAI_API_KEY or OPENAI_API_KEY env var")
        self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def _call_structured(self, schema: SchemaDict, user_query: str) -> Mapping[str, Any]:
        """Attempt structured JSON Schema API call with fallbacks."""
        client = self._ensure_client()
        payload_schema = self._INTENT_JSON_SCHEMA
        user_msg = self._USER_TEMPLATE.format(schema=json.dumps(schema, indent=2), query=user_query)

        try:
            chat = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._SYSTEM_TEMPLATE},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                response_format={"type": "json_schema", "json_schema": payload_schema},
            )
            content = chat.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            chat = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._SYSTEM_TEMPLATE},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
            )
            content = chat.choices[0].message.content or "{}"
            if content.startswith("```"):
                content = content.strip("`\n")
                if content.lower().startswith("json"):
                    content = content[4:].lstrip()
            try:
                return json.loads(content)
            except Exception as exc:
                raise AIQueryLLMError("LLM returned non-JSON") from exc

    def translate(self, schema: SchemaDict, user_query: str) -> IntentSpec:
        data = self._call_structured(schema=schema, user_query=user_query)
        return IntentSpec.from_json_dict(data)


DEFAULT_FLAT_LOOKUPS: Set[str] = {
    "exact",
    "iexact",
    "icontains",
    "istartswith",
    "iendswith",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "range",
    "isnull",
    "ne",
}


class IntentCompiler:
    """Validate an IntentSpec against the model and build a Q expression."""

    def __init__(
        self,
        model: Type[models.Model],
        *,
        depth: int = _DEFAULT_SCHEMA_DEPTH,
        include_reverse: bool = _DEFAULT_INCLUDE_REVERSE,
        allow_lookups: Mapping[str, Sequence[str]] | None = None,
        synonyms: Optional[Dict[str, Sequence[str]]] = None,
    ) -> None:
        self.model = model
        self.schema = build_schema_for_model(model, depth=depth, include_reverse=include_reverse, synonyms=synonyms)
        self.allow_lookups: Mapping[str, Sequence[str]] = allow_lookups or self._build_default_lookup_policy()

    def _build_default_lookup_policy(self) -> Mapping[str, Sequence[str]]:
        policy: Dict[str, Sequence[str]] = {}
        for f in self.model._meta.get_fields():
            # Handle concrete fields (forward relationships and regular fields)
            if getattr(f, "concrete", False):
                if getattr(f, "many_to_many", False) and not isinstance(f, models.ManyToManyField):
                    continue
                cat = _field_category(f)
                policy[f.name] = list(DEFAULT_TYPE_LOOKUPS.get(cat, {"exact"}))
            # Handle reverse relationships (non-concrete fields)
            elif getattr(f, "is_relation", False):
                # For reverse relationships, allow basic operations including isnull
                try:
                    # Get the accessor name safely
                    accessor_name = getattr(f, "get_accessor_name", lambda: None)()
                    if accessor_name:
                        # Different operations for different reverse relationship types
                        if hasattr(f, "many_to_many") and f.many_to_many:
                            # Reverse ManyToMany relationships
                            policy[accessor_name] = ["in", "isnull"]
                        else:
                            # Reverse ForeignKey and OneToOne relationships
                            policy[accessor_name] = ["exact", "in", "isnull"]
                except Exception:
                    # If we can't get accessor name, try the field name
                    if hasattr(f, "name") and f.name:
                        policy[f.name] = ["exact", "in", "isnull"]
        return policy

    def _resolve_path(self, path: FieldPath) -> Tuple[models.Field, str]:
        bits = path.split("__")
        model: Type[models.Model] = self.model
        field: Optional[models.Field] = None
        lookup_suffix: str = ""
        for i, bit in enumerate(bits):
            try:
                field = model._meta.get_field(bit)
            except Exception:
                lookup_suffix = "__".join(bits[i:])
                break
            if getattr(field, "is_relation", False) and i < len(bits) - 1:
                related_model = getattr(field, "related_model", None)
                if related_model:
                    model = related_model
                else:
                    lookup_suffix = "__".join(bits[i+1:])
                    break
            elif i < len(bits) - 1:
                lookup_suffix = "__".join(bits[i+1:])
                break
        if field is None:
            raise AIQueryModelMismatchError(path)
        return field, lookup_suffix

    def _validate_op(self, field: models.Field, op: str) -> str:
        op_lower = OP_ALIAS.get(op.lower(), op.lower())

        # For relationship traversal, be more permissive
        if getattr(field, "is_relation", False):
            # Allow basic operations on relationship fields
            allowed = {"exact", "iexact", "icontains", "in", "isnull", "gt", "gte", "lt", "lte", "ne"}
        else:
            # Use the configured lookup policy
            allowed = {o.lower() for o in self.allow_lookups.get(field.name, [])}
            if not allowed:
                allowed = {o.lower() for o in DEFAULT_TYPE_LOOKUPS.get(_field_category(field), {"exact"})}

        if op_lower not in allowed:
            raise AIQueryOperatorError(field.name, op_lower)
        return op_lower

    def _build_condition_q(self, cond: FilterSpec) -> Q:
        path = cond.path
        op = cond.op
        value = cond.value

        if not path:
            raise AIQueryModelMismatchError(str(path))

        bits = path.split("__")
        if bits[-1].lower() in DEFAULT_FLAT_LOOKUPS:
            op = bits[-1].lower()
            path = "__".join(bits[:-1])

        # For relationship traversal, we need to resolve the final field
        final_field, lookup_suffix = self._resolve_path(path)

        # If there's a lookup suffix, use the original path for the query
        # but validate against the final field
        if lookup_suffix:
            # This means we have a relationship traversal
            # The validation should be against the final field type
            op = self._validate_op(final_field, op)
        else:
            # Direct field access
            op = self._validate_op(final_field, op)

        if op == "ne":
            # Handle null values specially for negation
            if value is None:
                return Q(**{f"{path}__isnull": False})
            coerced = _coerce_value(final_field, value)
            return ~Q(**{f"{path}__exact": coerced})

        if op == "isnull":
            v = value
            if isinstance(v, str):
                v = v.strip().lower() in {"1", "true", "t", "yes", "y"}
            return Q(**{f"{path}__isnull": bool(v)})

        if op == "in":
            if not isinstance(value, (list, tuple)):
                value = [value]
            coerced = [_coerce_value(final_field, v) for v in value]
            return Q(**{f"{path}__in": coerced})

        if op == "range":
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise AIQueryValueError(final_field.name, value, msg="'range' expects [start,end]")
            start = _coerce_value(final_field, value[0])
            end = _coerce_value(final_field, value[1])
            return Q(**{f"{path}__range": (start, end)})

        coerced = _coerce_value(final_field, value)
        return Q(**{f"{path}__{op}": coerced})

    def compile(self, intent: IntentSpec) -> Q:
        if intent.status != "ok":
            raise AIQueryLLMError(intent.message or "LLM could not translate query.")
        q_total: Optional[Q] = None
        for cond in intent.filters:
            q = self._build_condition_q(cond)
            q_total = q if q_total is None else (q_total & q if intent.logic == "and" else q_total | q)
        return q_total or Q()


class YoloQuerySet(models.QuerySet):
    """QuerySet subclass with ai_query() powered by an LLM."""

    def ai_query(
        self,
        text: str,
        *,
        llm: Optional[LLMClient] = None,
        depth: Optional[int] = None,
        include_reverse: Optional[bool] = None,
        synonyms: Optional[Dict[str, Sequence[str]]] = None,
        allow_lookups: Mapping[str, Sequence[str]] | None = None,
        raise_errors: bool = False,
    ) -> models.QuerySet:
        llm = llm or default_llm()
        if depth is None:
            depth = get_default_schema_depth()
        if include_reverse is None:
            include_reverse = get_default_include_reverse()

        schema = build_schema_for_model(self.model, depth=depth, include_reverse=include_reverse, synonyms=synonyms)
        try:
            intent = llm.translate(schema=schema, user_query=text)
            compiler = IntentCompiler(
                self.model, depth=depth, include_reverse=include_reverse, allow_lookups=allow_lookups, synonyms=synonyms
            )
            q = compiler.compile(intent)
        except AIQueryError as e:
            if raise_errors:
                raise
            qs = self.none()
            qs.ai_error = e
            return qs

        qs = self.filter(q)
        if intent.order_by:
            try:
                qs = qs.order_by(*intent.order_by)
            except Exception as e:
                if raise_errors:
                    raise AIQueryModelMismatchError(f"Invalid ordering field: {e}")
                qs = self.none()
                qs.ai_error = AIQueryModelMismatchError(f"Invalid ordering field: {e}")
                return qs
        if intent.limit is not None and isinstance(intent.limit, int) and intent.limit >= 0:
            qs = qs[: intent.limit]
        return qs


class YoloManager(models.Manager):
    """Manager that returns YoloQuerySet and exposes ai_query()."""

    def get_queryset(self) -> YoloQuerySet:
        return YoloQuerySet(self.model, using=self._db)

    def ai_query(self, *args: Any, **kwargs: Any) -> models.QuerySet:
        return self.get_queryset().ai_query(*args, **kwargs)


AIManager = YoloManager
AIQuerySet = YoloQuerySet
_default_llm_singleton: Optional[LLMClient] = None


def default_llm() -> LLMClient:
    global _default_llm_singleton
    if _default_llm_singleton is None:
        _default_llm_singleton = LLMClient(model_name=get_openai_model_name())
    return _default_llm_singleton


def _make_wrapped_manager(model: Type[models.Model], base_manager: models.Manager) -> YoloManager:
    """Create a YoloManager subclass that also inherits the original manager's class."""
    base_cls = base_manager.__class__

    if isinstance(base_manager, YoloManager):
        return base_manager

    if issubclass(base_cls, YoloManager):
        mgr = base_cls()
        mgr.model = model
        return mgr

    class _WrappedYoloManager(YoloManager, base_cls):
        pass

    mgr = _WrappedYoloManager()
    mgr.model = model
    return mgr


def _install_on_model(model: Type[models.Model]) -> None:
    if getattr(model, "_yoloquery_installed", False):
        return
    base_manager = model._meta.default_manager
    wrapped = _make_wrapped_manager(model, base_manager)

    model._orig_objects = base_manager
    model.objects = wrapped
    model.yolo = wrapped
    model._meta.default_manager = wrapped

    logger.warning("YOLOQuery overriding .objects on %s.%s", model._meta.app_label, model.__name__)
    model._yoloquery_installed = True


def auto_install_yolo_managers() -> None:
    """Install YoloManager across configured models based on settings patterns."""
    pats = get_auto_install_patterns()
    if not pats:
        return
    all_configs = {cfg.label: cfg for cfg in django_apps.get_app_configs()}

    def _match(app_label: str, model_name: str) -> bool:
        for p in pats:
            if p in {"*", "*.*"}:
                return True
            if "." in p:
                a, m = p.split(".", 1)
                if a == app_label and (m == "*" or m.lower() == model_name.lower()):
                    return True
            else:
                if p == app_label:
                    return True
        return False

    for cfg in all_configs.values():
        for model in cfg.get_models():
            if _match(cfg.label, model.__name__):
                _install_on_model(model)


from django.apps import AppConfig  # noqa: E402


class DjangoYoloQueryAppConfig(AppConfig):
    name = "django_yoloquery"
    verbose_name = "Django YOLOQuery"

    def ready(self) -> None:
        try:
            auto_install_yolo_managers()
        except Exception:
            logger.exception("YOLOQuery auto-install failed")


default_app_config = "django_yoloquery.DjangoYoloQueryAppConfig"


class DummyLLM(LLMClient):
    """Test double returning canned IntentSpecs."""

    def __init__(self, responses: Mapping[str, Mapping[str, Any]]):
        super().__init__(model_name="dummy", client=None)
        self.responses = responses

    def translate(self, schema: SchemaDict, user_query: str) -> IntentSpec:
        data = self.responses.get(user_query)
        if data is None:
            data = {"status": "ok", "filters": []}
        return IntentSpec.from_json_dict(dict(data))


__all__ = [
    "YoloQuerySet",
    "YoloManager",
    "LLMClient",
    "IntentCompiler",
    "FilterSpec",
    "IntentSpec",
    "AIQueryError",
    "AIQueryModelMismatchError",
    "AIQueryOperatorError",
    "AIQueryValueError",
    "AIQueryLLMError",
    "build_schema_for_model",
    "default_llm",
    "auto_install_yolo_managers",
    "DummyLLM",
    "AIManager",
    "AIQuerySet",
    "get_default_schema_depth",
    "get_default_include_reverse",
    "get_auto_install_patterns",
    "get_openai_model_name",
    "get_openai_api_key",
]
