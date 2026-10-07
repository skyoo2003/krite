"""Jev wire conformance: the official TypeSafe Python SDK, pointed at a Krite server by `base_url` only.

Checks shapes, keys, and sums, never answer values, so it runs against any model: the weight-free
`hash_server` example in CI (`scripts/check-compat.sh`) or a real `krite serve`.
Set `KRITE_BASE_URL` to the server root, e.g. `http://127.0.0.1:8110`.
"""

import asyncio
import math
import os

import pytest
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeClient,
    TypeSafeUnprocessableEntityError,
)

BASE_URL = os.environ.get("KRITE_BASE_URL")
pytestmark = pytest.mark.skipif(not BASE_URL, reason="set KRITE_BASE_URL to a running Krite server")

JEV_ALIAS = "jev-latest"
STATE = "Hi, I was charged twice for my March subscription."
LEVELS = ["Can wait", {"level": "Needs attention this week"}, "Needs attention today"]


def options() -> dict:
    # Any key works: Krite has no auth, and the SDK requires a key client-side. No retries: a 5xx fails at once.
    return {"api_key": "krite", "base_url": BASE_URL, "retry": RetryPolicy(max_retries=0)}


@pytest.fixture(scope="module")
def client():
    with TypeSafeClient(**options()) as c:
        yield c


@pytest.fixture(scope="module")
def served(client) -> str:
    return next(m.name for m in client.models.list().models if m.name != JEV_ALIAS)


def route(**kw) -> Choice:
    return Choice(criteria={"billing": "Charges and refunds", "technical": None, "other": None}, **kw)


def assert_distribution(probs: dict) -> None:
    assert all(0 <= p <= 1 for p in probs.values())
    assert math.isclose(sum(probs.values()), 1, abs_tol=1e-6)


def test_models_list_has_the_served_model_and_jev_alias(client, served):
    models = client.models.list().models
    assert {served, JEV_ALIAS} <= {m.name for m in models}
    assert all(isinstance(m.description, str) and isinstance(m.release_date, str) for m in models)


def test_default_model_alias_resolves_to_the_served_model(client, served):
    r = client.system_one(state=STATE, questions={"route": route(instructions="Which team?")})
    assert r.model == served
    assert r.request_id
    assert r.usage.input_tokens > 0 and r.usage.output_tokens == 0


def test_explicit_model(client, served):
    r = client.system_one(state=STATE, questions={"route": route(instructions="Which team?")}, model=served)
    assert r.model == served


def test_choice(client):
    a = client.system_one(state=STATE, questions={"route": route(instructions="Which team?")}).choices["route"]
    assert a.choice in {"billing", "technical", "other"}
    assert set(a.probabilities) == {"billing", "technical", "other"}
    assert_distribution(a.probabilities)
    assert a.probabilities[a.choice] == max(a.probabilities.values())
    assert 0 <= a.confidence <= 1


def test_score_is_keyed_by_level_index(client):
    q = Score(instructions="How urgent is this?", criteria=LEVELS)
    a = client.system_one(state=STATE, questions={"urgency": q}).scores["urgency"]
    assert list(a.legend) == list(a.probabilities) == [0, 1, 2]
    assert [a.legend[i] for i in range(3)] == LEVELS
    assert_distribution(a.probabilities)
    assert math.isclose(a.score, sum(i * p for i, p in a.probabilities.items()), abs_tol=1e-9)
    assert 0 <= a.confidence <= 1


def test_noul(client):
    a = client.system_one(state=STATE, questions={"refund": Noul(instructions="Is a refund requested?")}).nouls
    assert 0 <= a["refund"].noul <= 1


def test_sdk_legal_request_shapes(client):
    questions = {
        "no_instructions": route(),
        "json_instructions": Score(instructions={"task": "Rate urgency", "scale": "0-2"}, criteria=LEVELS),
        "json_criteria": Noul(instructions="Refund?", criteria={"true": {"asks": "money back"}, "false": None}),
        "dict_form": {"type": "noul", "instructions": "Is the customer angry?"},
    }
    for state in [STATE, {"subject": "Duplicate charge", "body": STATE}, [{"from": "customer", "text": STATE}]]:
        r = client.system_one(state=state, questions=questions)
        assert set(r.answers) == set(questions)
        assert (set(r.choices), set(r.scores)) == ({"no_instructions"}, {"json_instructions"})


def test_object_state_key_order_does_not_matter(client):
    questions = {"route": route(instructions="Which team?"), "urgency": Score(criteria=LEVELS)}
    a = client.system_one(state={"subject": "Charge", "body": STATE}, questions=questions)
    b = client.system_one(state={"body": STATE, "subject": "Charge"}, questions=questions)
    assert a.answers == b.answers


def test_async_client(served):
    async def call():
        async with AsyncTypeSafeClient(**options()) as c:
            return await c.system_one(state=STATE, questions={"route": route(instructions="Which team?")})

    assert asyncio.run(call()).model == served


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"questions": {"route": route(instructions="Which team?")}, "model": "no-such-model"}, "unknown_model"),
        ({"questions": {f"q{i}": Noul(instructions="Yes?") for i in range(65)}}, "too_many_questions"),
        ({"questions": {"route": Choice(instructions="Which team?", criteria={})}}, "invalid_request"),
    ],
)
def test_errors_reach_the_sdk_as_422(client, kwargs, error_type):
    with pytest.raises(TypeSafeUnprocessableEntityError) as e:
        client.system_one(state=STATE, **kwargs)
    assert e.value.status == 422
    assert e.value.body["error"]["type"] == error_type
    assert e.value.request_id
