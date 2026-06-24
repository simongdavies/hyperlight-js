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

"""Tests for the `LoadedJSSandbox.eval` REPL primitive.

VM-booting tests are skipped when no hypervisor is available.
"""

import hyperlight_js as hl
import pytest

requires_hypervisor = pytest.mark.skipif(
    not hl.is_hypervisor_present(),
    reason="no hypervisor available (KVM/WHP/mshv)",
)


def _eval_sandbox():
    """A loaded sandbox with no handlers, ready for `eval()` (REPL use)."""
    proto = hl.SandboxBuilder().build()
    sandbox = proto.load_runtime()
    return sandbox.get_loaded_sandbox()


# ── Non-VM tests (input validation) ───────────────────────────────────


def test_eval_method_exists():
    assert hasattr(hl.LoadedJSSandbox, "eval")


# ── VM tests ──────────────────────────────────────────────────────────


@requires_hypervisor
def test_eval_expression():
    s = _eval_sandbox()
    assert s.eval("1 + 2") == 3


@requires_hypervisor
def test_eval_object_marshals():
    s = _eval_sandbox()
    assert s.eval("({ a: 1, b: [2, 3] })") == {"a": 1, "b": [2, 3]}


@requires_hypervisor
def test_eval_undefined_is_none():
    s = _eval_sandbox()
    assert s.eval("undefined") is None


@requires_hypervisor
def test_eval_function_is_none():
    s = _eval_sandbox()
    assert s.eval("(function () {})") is None


@requires_hypervisor
def test_eval_let_const_persist():
    s = _eval_sandbox()
    s.eval("let a = 10; const b = 5;")
    assert s.eval("a + b") == 15


@requires_hypervisor
def test_eval_function_declaration_persists():
    s = _eval_sandbox()
    s.eval("function sq(x) { return x * x; }")
    assert s.eval("sq(9)") == 81


@requires_hypervisor
def test_eval_mutated_state_accumulates():
    s = _eval_sandbox()
    s.eval("let n = 0;")
    s.eval("n += 2;")
    s.eval("n += 3;")
    assert s.eval("n") == 5


@requires_hypervisor
def test_eval_resolves_promise():
    s = _eval_sandbox()
    assert s.eval("Promise.resolve(42)") == 42


@requires_hypervisor
def test_eval_shares_state_with_handler():
    proto = hl.SandboxBuilder().build()
    sandbox = proto.load_runtime()
    sandbox.add_handler("read", "function handler() { return globalThis.shared; }")
    loaded = sandbox.get_loaded_sandbox()
    loaded.eval("globalThis.shared = 123;")
    assert loaded.call_handler("read", {}) == 123


@requires_hypervisor
def test_eval_throw_raises():
    s = _eval_sandbox()
    with pytest.raises(hl.HyperlightError) as exc_info:
        s.eval("throw new Error('kaboom')")
    assert "kaboom" in str(exc_info.value)


@requires_hypervisor
def test_eval_rejects_empty_code():
    # Validation is pure-Python, but obtaining a LoadedJSSandbox boots a VM.
    proto = hl.SandboxBuilder().build()
    sandbox = proto.load_runtime()
    loaded = sandbox.get_loaded_sandbox()
    with pytest.raises(hl.InvalidArgError):
        loaded.eval("")


@requires_hypervisor
def test_eval_rejects_bad_timeout():
    proto = hl.SandboxBuilder().build()
    sandbox = proto.load_runtime()
    loaded = sandbox.get_loaded_sandbox()
    with pytest.raises(hl.InvalidArgError):
        loaded.eval("1", wall_clock_timeout_ms=0)


@requires_hypervisor
def test_eval_with_monitor_succeeds():
    s = _eval_sandbox()
    assert s.eval("6 * 7", wall_clock_timeout_ms=5000) == 42


@requires_hypervisor
def test_eval_with_monitor_times_out():
    s = _eval_sandbox()
    with pytest.raises(hl.CancelledError):
        s.eval("while (true) {}", wall_clock_timeout_ms=200)
    assert s.poisoned is True
