# Copyright 2026  The Hyperlight Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the `hyperlight_js` Python host binding.

Tests that boot a micro-VM are skipped when no hypervisor is available
(`is_hypervisor_present()` is `False`), so this module imports and the
non-VM tests still run on machines without virtualization.
"""

import threading
import time

import hyperlight_js as hl
import pytest

requires_hypervisor = pytest.mark.skipif(
    not hl.is_hypervisor_present(),
    reason="no hypervisor available (KVM/WHP/mshv)",
)


def _loaded(handler_name, script, *, host_modules=None, modules=None):
    """Build a loaded sandbox with a single handler.

    Args:
        handler_name: Name to register the handler under.
        script: Handler source (must define `function handler(event)`).
        host_modules: Optional list of `(module, fn_name, callable)` host fns.
        modules: Optional list of `(name, source)` user ES modules.

    Returns:
        A `LoadedJSSandbox` ready for `call_handler`.
    """
    proto = hl.SandboxBuilder().build()
    for module, fn_name, fn in host_modules or []:
        proto.host_module(module).register(fn_name, fn)
    sandbox = proto.load_runtime()
    for name, source in modules or []:
        sandbox.add_module(name, source)
    sandbox.add_handler(handler_name, script)
    return sandbox.get_loaded_sandbox()


# ── Non-VM tests (input validation, run without a hypervisor) ──────────


def test_exception_hierarchy():
    for exc in (
        hl.PoisonedError,
        hl.CancelledError,
        hl.GuestAbortError,
        hl.InvalidArgError,
        hl.ConsumedError,
        hl.InternalError,
    ):
        assert issubclass(exc, hl.HyperlightError)


def test_error_codes_are_exposed():
    assert hl.CancelledError.code == "ERR_CANCELLED"
    assert hl.PoisonedError.code == "ERR_POISONED"
    assert hl.GuestAbortError.code == "ERR_GUEST_ABORT"
    assert hl.InvalidArgError.code == "ERR_INVALID_ARG"
    assert hl.ConsumedError.code == "ERR_CONSUMED"
    assert hl.InternalError.code == "ERR_INTERNAL"


def test_add_handler_empty_name_rejected():
    proto = hl.SandboxBuilder().build()
    sandbox = proto.load_runtime()
    with pytest.raises(hl.InvalidArgError):
        sandbox.add_handler("", "function handler(e){ return 1; }")


def test_reserved_namespace_rejected():
    proto = hl.SandboxBuilder().build()
    sandbox = proto.load_runtime()
    with pytest.raises(hl.InvalidArgError):
        sandbox.add_module("math", "export const x = 1;", "host")


# ── VM tests ───────────────────────────────────────────────────────────


@requires_hypervisor
def test_basic_handler_eval():
    loaded = _loaded(
        "run",
        "function handler(e){ return { doubled: e.n * 2, who: e.name }; }",
    )
    result = loaded.call_handler("run", {"n": 21, "name": "ada"}, cpu_timeout_ms=2000)
    assert result == {"doubled": 42, "who": "ada"}


@requires_hypervisor
def test_state_persists_across_calls():
    loaded = _loaded(
        "step",
        "function handler(e){ globalThis.n = (globalThis.n||0) + e.by; return globalThis.n; }",
    )
    assert loaded.call_handler("step", {"by": 5}) == 5
    assert loaded.call_handler("step", {"by": 7}) == 12
    assert loaded.call_handler("step", {"by": 0}) == 12


@requires_hypervisor
def test_host_function_sync():
    loaded = _loaded(
        "run",
        'import * as math from "host:math";\n'
        "function handler(e){ return math.add(e.a, e.b); }",
        host_modules=[("math", "add", lambda a, b: a + b)],
    )
    assert loaded.call_handler("run", {"a": 2, "b": 40}) == 42


@requires_hypervisor
def test_host_function_binary_roundtrip():
    received = {}

    def echo(data):
        received["arg"] = bytes(data)
        return bytes(reversed(data))

    loaded = _loaded(
        "run",
        'import * as b from "host:b";\n'
        "function handler(e){ const out = b.echo(new Uint8Array([1,2,3])); return Array.from(out); }",
        host_modules=[("b", "echo", echo)],
    )
    assert loaded.call_handler("run", {}) == [3, 2, 1]
    assert received["arg"] == b"\x01\x02\x03"


@requires_hypervisor
def test_user_es_module():
    loaded = _loaded(
        "run",
        "import { add } from 'user:math';\n"
        "function handler(e){ return add(e.a, e.b); }",
        modules=[("math", "export function add(a, b){ return a + b; }")],
    )
    assert loaded.call_handler("run", {"a": 19, "b": 23}) == 42


@requires_hypervisor
def test_host_print_capture():
    lines = []
    proto = hl.SandboxBuilder().host_print_fn(lines.append).build()
    sandbox = proto.load_runtime()
    sandbox.add_handler("run", "function handler(e){ console.log('hello ' + e.who); return 1; }")
    loaded = sandbox.get_loaded_sandbox()
    loaded.call_handler("run", {"who": "world"})
    assert any("hello world" in line for line in lines)


@requires_hypervisor
def test_snapshot_restore():
    loaded = _loaded(
        "step",
        "function handler(e){ globalThis.n = (globalThis.n||0) + e.by; return globalThis.n; }",
    )
    assert loaded.call_handler("step", {"by": 5}) == 5
    snap = loaded.snapshot()
    assert loaded.call_handler("step", {"by": 10}) == 15
    loaded.restore(snap)
    assert loaded.call_handler("step", {"by": 0}) == 5


@requires_hypervisor
def test_wall_clock_timeout_raises_cancelled():
    loaded = _loaded("spin", "function handler(e){ while(true){} }")
    with pytest.raises(hl.CancelledError):
        loaded.call_handler("spin", {}, wall_clock_timeout_ms=50)


@requires_hypervisor
def test_interrupt_kill():
    loaded = _loaded("spin", "function handler(e){ while(true){} }")
    handle = loaded.interrupt_handle
    timer = threading.Timer(0.2, handle.kill)
    timer.start()
    try:
        with pytest.raises(hl.CancelledError):
            loaded.call_handler("spin", {})
    finally:
        timer.cancel()


@requires_hypervisor
def test_guest_throw_raises_hyperlight_error():
    loaded = _loaded("boom", "function handler(e){ throw new Error('kaboom'); }")
    with pytest.raises(hl.HyperlightError) as info:
        loaded.call_handler("boom", {})
    assert info.value.__class__.code.startswith("ERR_")


@requires_hypervisor
def test_call_stats_populated():
    loaded = _loaded("run", "function handler(e){ return 1; }")
    assert loaded.last_call_stats is None
    loaded.call_handler("run", {}, cpu_timeout_ms=2000)
    stats = loaded.last_call_stats
    assert stats is not None
    assert stats.wall_clock_ms >= 0.0


@requires_hypervisor
def test_consumed_after_unload():
    loaded = _loaded("run", "function handler(e){ return 1; }")
    loaded.unload()
    with pytest.raises(hl.ConsumedError):
        loaded.call_handler("run", {})
