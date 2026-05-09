# Testing

## Framework

pytest.

## Running

```bash
uv run pytest tests/
uv run pytest tests/ -v -k <name>  # filter by name
```

## Conventions

- Test files: `test_*.py` or `*_test.py`
- Mirror module structure: `tests/<module>/test_<thing>.py`
- One assertion focus per test when practical
- Fixtures in `conftest.py`, not duplicated across files

## Mocking

- Prefer real implementations
- Mock only at boundaries: network, time, randomness, filesystem

## Anti-patterns

- Don't mock what you can test directly
- Don't test implementation details (private methods, internal state)
- Don't share mutable state between tests
- Don't write tests that pass when production code is broken
