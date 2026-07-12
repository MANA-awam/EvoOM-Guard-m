# Black-box HTTP recipe — judge a service over the wire

The judge-owned pack never imports the candidate. It launches the HTTP service
**across a process boundary** via `$EVOGUARD_EXEC`, waits for the port, asserts
on the **observable HTTP responses**, and tears it down. Forgery code in the
candidate (fake `3 passed` on stdout, `atexit` + `os._exit(0)`) runs inside the
*server* process — it cannot reach the judge, and only responses are asserted.

```bash
cd examples/blackbox-http

# Honest fix: /add finally returns a+b  ->  PASS (pack 1/1 AND repo suite)
evo-guard guard ./sample_repo --patch patches/honest.txt \
    --verifier-pack ./pack --blackbox

# Forgery: bug kept, fake pass printed, exit forced 0  ->  FAIL (pack 0/1)
evo-guard guard ./sample_repo --patch patches/cheat.txt \
    --verifier-pack ./pack --blackbox
```

## The boundary, honestly

This recipe needs a judge↔candidate **localhost channel**, so it runs under the
**subprocess** black-box boundary (`report_integrity:
external_process_isolated`, `candidate_isolation: subprocess`). The hardened
container modes default to `--network none`, which deliberately severs that
channel — pass a docker network only if you consciously accept the trade-off.
For maximum isolation without networking, wrap the behaviour behind a CLI entry
point instead (see [`../blackbox-cli/`](../blackbox-cli/)).

## Layout

```
sample_repo/   the service under judgment (stdlib HTTP server + its own tests)
pack/          judge-owned protocol tests + pack.json (a versioned behaviour contract)
patches/       honest fix and an in-process forgery attempt
```
