# krite

The [Krite](https://github.com/skyoo2003/krite) decision runtime in process: Protocol v1 answers from Python, with the same runtime, the same bytes, and the same errors as `krite serve`, and no server.

```bash
pip install krite
hf download skyoo2003/krite-0.15b-v1 --local-dir krite-0.15b-v1
```

```python
import krite

k = krite.Krite("krite-0.15b-v1")
r = k.decide(
    {
        "state": [
            {"from": "customer", "text": "Hi, I was charged twice for my March subscription."},
            {"from": "agent", "text": "Sorry about that. Can you share the invoice number?"},
        ],
        "questions": {
            "route": {"type": "choice", "criteria": {"billing": None, "technical": None, "other": None}},
            "refund_requested": {"type": "noul"},
        },
    }
)
print(r["answers"]["route"]["choice"], r["answers"]["refund_requested"]["noul"])
```

`decide` accepts a dict, a JSON string, or JSON bytes and returns the response as a dict. The model alias `jev-latest` resolves to the loaded model, as on the server.

| Error | `status` | `type` |
|---|---|---|
| Malformed JSON, schema violation, body over 4 MiB | 422 | `invalid_request` |
| Model other than the loaded one or `jev-latest` | 422 | `unknown_model` |
| Over a server limit | 422 | `state_too_long`, `too_many_questions`, `too_many_options` |
| Runtime failure | 500 | `internal_error` |

Errors raise `krite.KriteError` with `status`, `type`, `message`, and `param`. A missing or broken model directory raises `RuntimeError` when `Krite` is constructed.

`device="auto"` uses Metal on Apple silicon and the CPU elsewhere; `device="cpu"` forces the CPU. Calls run one at a time on one `Krite` and release the GIL, so threads may share it. See [docs/runtime.md](https://github.com/skyoo2003/krite/blob/main/docs/runtime.md#python).
