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

//! In-process tests for the `JsRuntime::eval` REPL primitive.
//!
//! These exercise the eval semantics directly against the runtime (no
//! Hyperlight VM), verifying that arbitrary JavaScript runs against the
//! persistent global context and that top-level declarations persist across
//! calls — the property that makes a `quickjs`-style REPL possible.

#![cfg(not(hyperlight))]

/// A minimal Host that doesn't support loading external modules.
struct NoOpHost;

impl hyperlight_js_runtime::host::Host for NoOpHost {
    fn resolve_module(&self, _base: String, name: String) -> anyhow::Result<String> {
        anyhow::bail!("NoOpHost does not support resolving module '{name}'")
    }

    fn load_module(&self, name: String) -> anyhow::Result<String> {
        anyhow::bail!("NoOpHost does not support loading module '{name}'")
    }
}

fn new_runtime() -> hyperlight_js_runtime::JsRuntime {
    hyperlight_js_runtime::JsRuntime::new(NoOpHost).expect("Failed to create JsRuntime")
}

// Provide the module/global init symbols this test binary links against.
hyperlight_js_runtime::native_modules! {}
hyperlight_js_runtime::custom_globals! {}

#[test]
fn eval_returns_expression_value() {
    let mut rt = new_runtime();
    let out = rt.eval("1 + 2".to_string(), false).unwrap();
    assert_eq!(out, "3");
}

#[test]
fn eval_returns_object_as_json() {
    let mut rt = new_runtime();
    let out = rt.eval("({ a: 1, b: [2, 3] })".to_string(), false).unwrap();
    assert_eq!(out, r#"{"a":1,"b":[2,3]}"#);
}

#[test]
fn eval_undefined_becomes_null() {
    let mut rt = new_runtime();
    // A bare statement has an `undefined` completion value.
    let out = rt.eval("undefined".to_string(), false).unwrap();
    assert_eq!(out, "null");
}

#[test]
fn eval_function_value_becomes_null() {
    let mut rt = new_runtime();
    // Functions are not JSON-serializable; surfaced as `null`, not an error.
    let out = rt.eval("(function () {})".to_string(), false).unwrap();
    assert_eq!(out, "null");
}

#[test]
fn eval_var_persists_across_calls() {
    let mut rt = new_runtime();
    rt.eval("var x = 10;".to_string(), false).unwrap();
    let out = rt.eval("x + 5".to_string(), false).unwrap();
    assert_eq!(out, "15");
}

#[test]
fn eval_let_persists_across_calls() {
    let mut rt = new_runtime();
    rt.eval("let y = 21;".to_string(), false).unwrap();
    let out = rt.eval("y * 2".to_string(), false).unwrap();
    assert_eq!(out, "42");
}

#[test]
fn eval_const_persists_across_calls() {
    let mut rt = new_runtime();
    rt.eval("const z = 7;".to_string(), false).unwrap();
    let out = rt.eval("z * z".to_string(), false).unwrap();
    assert_eq!(out, "49");
}

#[test]
fn eval_function_declaration_persists() {
    let mut rt = new_runtime();
    rt.eval("function add(a, b) { return a + b; }".to_string(), false)
        .unwrap();
    let out = rt.eval("add(3, 4)".to_string(), false).unwrap();
    assert_eq!(out, "7");
}

#[test]
fn eval_mutated_state_persists() {
    let mut rt = new_runtime();
    rt.eval("let counter = 0;".to_string(), false).unwrap();
    rt.eval("counter += 1;".to_string(), false).unwrap();
    rt.eval("counter += 1;".to_string(), false).unwrap();
    let out = rt.eval("counter".to_string(), false).unwrap();
    assert_eq!(out, "2");
}

#[test]
fn eval_resolves_immediate_promise() {
    let mut rt = new_runtime();
    let out = rt.eval("Promise.resolve(99)".to_string(), false).unwrap();
    assert_eq!(out, "99");
}

#[test]
fn eval_await_resolves() {
    let mut rt = new_runtime();
    let out = rt
        .eval("(async () => 5)().then(v => v + 1)".to_string(), false)
        .unwrap();
    assert_eq!(out, "6");
}

#[test]
fn eval_throw_is_error() {
    let mut rt = new_runtime();
    let err = rt
        .eval("throw new Error('boom')".to_string(), false)
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("boom"),
        "error should surface the thrown message, got: {err:#}"
    );
}

#[test]
fn eval_state_shared_with_handler() {
    let mut rt = new_runtime();
    // State defined via eval should be visible to a registered handler.
    rt.eval("globalThis.shared = 123;".to_string(), false)
        .unwrap();
    rt.register_handler(
        "read_shared",
        "function handler() { return globalThis.shared; }",
        ".",
    )
    .unwrap();
    let out = rt
        .run_handler("read_shared".to_string(), "{}".to_string(), false)
        .unwrap();
    assert_eq!(out, "123");
}
