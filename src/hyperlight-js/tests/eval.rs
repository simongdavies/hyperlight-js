/*
Copyright 2026  The Hyperlight Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
//! Test the `eval` REPL primitive end-to-end through a real Hyperlight VM.

#![allow(clippy::disallowed_macros)]

#[cfg(feature = "monitor-wall-clock")]
use std::time::Duration;

use hyperlight_js::SandboxBuilder;
#[cfg(feature = "monitor-wall-clock")]
use hyperlight_js::WallClockMonitor;

/// Build a loaded sandbox ready for `eval` calls. No handler is registered —
/// `eval` does not depend on one.
fn loaded() -> hyperlight_js::LoadedJSSandbox {
    let proto = SandboxBuilder::new().build().unwrap();
    let sandbox = proto.load_runtime().unwrap();
    sandbox.get_loaded_sandbox().unwrap()
}

#[test]
fn eval_expression() {
    let mut s = loaded();
    let out = s.eval("1 + 2".to_string(), None).unwrap();
    assert_eq!(out, "3");
}

#[test]
fn eval_object_marshals_to_json() {
    let mut s = loaded();
    let out = s.eval("({ a: 1, b: [2, 3] })".to_string(), None).unwrap();
    assert_eq!(out, r#"{"a":1,"b":[2,3]}"#);
}

#[test]
fn eval_undefined_is_null() {
    let mut s = loaded();
    assert_eq!(s.eval("undefined".to_string(), None).unwrap(), "null");
}

#[test]
fn eval_let_const_persist_across_calls() {
    let mut s = loaded();
    s.eval("let a = 10; const b = 5;".to_string(), None)
        .unwrap();
    let out = s.eval("a + b".to_string(), None).unwrap();
    assert_eq!(out, "15");
}

#[test]
fn eval_function_declaration_persists() {
    let mut s = loaded();
    s.eval("function sq(x) { return x * x; }".to_string(), None)
        .unwrap();
    assert_eq!(s.eval("sq(9)".to_string(), None).unwrap(), "81");
}

#[test]
fn eval_mutated_state_accumulates() {
    let mut s = loaded();
    s.eval("let n = 0;".to_string(), None).unwrap();
    s.eval("n += 2;".to_string(), None).unwrap();
    s.eval("n += 3;".to_string(), None).unwrap();
    assert_eq!(s.eval("n".to_string(), None).unwrap(), "5");
}

#[test]
fn eval_resolves_immediate_promise() {
    let mut s = loaded();
    assert_eq!(
        s.eval("Promise.resolve(42)".to_string(), None).unwrap(),
        "42"
    );
}

#[test]
fn eval_throw_is_guest_error() {
    let mut s = loaded();
    let err = s
        .eval("throw new Error('kaboom')".to_string(), None)
        .unwrap_err();
    assert!(
        err.to_string().contains("kaboom"),
        "expected thrown message in error, got: {err}"
    );
}

#[test]
fn eval_rejects_empty_code() {
    let mut s = loaded();
    assert!(s.eval(String::new(), None).is_err());
}

#[cfg(feature = "monitor-wall-clock")]
#[test]
fn eval_with_monitor_succeeds_under_limit() {
    let mut s = loaded();
    let monitor = WallClockMonitor::new(Duration::from_secs(5)).unwrap();
    let out = s
        .eval_with_monitor("6 * 7".to_string(), &monitor, None)
        .unwrap();
    assert_eq!(out, "42");
}

#[cfg(feature = "monitor-wall-clock")]
#[test]
fn eval_with_monitor_times_out_infinite_loop() {
    let mut s = loaded();
    let monitor = WallClockMonitor::new(Duration::from_millis(200)).unwrap();
    let res = s.eval_with_monitor("while (true) {}".to_string(), &monitor, None);
    assert!(
        res.is_err(),
        "infinite loop should be terminated by monitor"
    );
    assert!(s.poisoned(), "sandbox should be poisoned after termination");
}
