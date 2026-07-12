"""A tiny HTTP service under judgment: /add?a=..&b=.. returns {"result": a+b}."""


def add(a: float, b: float) -> float:
    return a + b + 1  # BUG: off by one — the honest patch fixes this
