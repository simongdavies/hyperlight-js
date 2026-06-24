# hyperlight-js (Python host binding)

Python host bindings for [`hyperlight-js`](https://github.com/hyperlight-dev/hyperlight-js):
run JavaScript inside a [Hyperlight](https://github.com/hyperlight-dev/hyperlight)
micro-VM (hardware-isolated sandbox) from Python.

This crate mirrors the Node/NAPI `js-host-api` binding, exposing the same staged
sandbox lifecycle:

```
SandboxBuilder → build() → ProtoJSSandbox → load_runtime()
  → JSSandbox → get_loaded_sandbox() → LoadedJSSandbox
```

## Requirements

- A supported hypervisor: KVM (Linux), WHP (Windows), or mshv. Check at runtime
  with `hyperlight_js.is_hypervisor_present()`.
- Building from source needs the Hyperlight guest toolchain (clang + the
  `cargo-hyperlight` cross-compile target), since the runtime is built into the
  `hyperlight-js` crate.

## Build

Built with [maturin](https://www.maturin.rs/):

```bash
# From this directory (src/py-host-api):
maturin develop            # build + install into the active venv
maturin build --release    # produce an abi3 wheel (CPython 3.9+)
```

## Example

```python
import hyperlight_js as hl

proto = hl.SandboxBuilder().build()

# Register a host function callable from guest JS as `import * as m from "host:math"`.
proto.host_module("math").register("add", lambda a, b: a + b)

sandbox = proto.load_runtime()
sandbox.add_handler(
    "run",
    "function handler(e) { return { sum: math.add(e.a, e.b) }; }",
)
loaded = sandbox.get_loaded_sandbox()

result = loaded.call_handler("run", {"a": 2, "b": 3}, cpu_timeout_ms=500)
assert result == {"sum": 5}
```

Host functions are **synchronous** Python callables. `bytes` / `bytearray`
arguments and return values cross the boundary as binary; everything else is
JSON. Errors are raised as typed exceptions (`PoisonedError`, `CancelledError`,
`GuestAbortError`, `InvalidArgError`, `ConsumedError`, `InternalError`) under the
common base `HyperlightError`.
