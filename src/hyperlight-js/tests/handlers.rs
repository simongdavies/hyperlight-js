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
//! Test the behaviour of JavaScript handlers

#![allow(clippy::disallowed_macros)]

use hyperlight_js::{SandboxBuilder, Script};

#[test]
fn handle_event() {
    let handler = Script::from_content(
        r#"
        function handler(event) {
            event.result = "Hello, " + event.name + "!";
            return event
        }
        "#,
    );

    let event = r#"
    {
        "name": "world",
        "result": ""
    }"#;

    let proto_js_sandbox = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto_js_sandbox.load_runtime().unwrap();

    sandbox.add_handler("handler", handler).unwrap();

    let mut loaded_sandbox = sandbox.get_loaded_sandbox().unwrap();

    let res = loaded_sandbox.handle_event("handler", event.to_string(), None);
    assert!(res.is_ok());

    let res = res.unwrap();
    assert_eq!(res, r#"{"name":"world","result":"Hello, world!"}"#);
}

#[test]
fn check_javascript_handler_returns_value() {
    let handler = Script::from_content(
        r#"
        function handler(event) {
            1
        }
        "#,
    );

    let event = r#"{}"#;

    let proto_js_sandbox = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto_js_sandbox.load_runtime().unwrap();

    sandbox.add_handler("handler", handler).unwrap();

    let mut loaded_sandbox = sandbox.get_loaded_sandbox().unwrap();

    let res = loaded_sandbox.handle_event("handler", event.to_string(), None);

    assert!(res.is_err());

    let err = res.unwrap_err();

    assert_eq!(
        err.to_string(),
        "Guest error occurred GuestError: Error: The handler function did not return a value"
    );
}

#[test]
fn add_handler_rejects_empty_name() {
    let proto = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto.load_runtime().unwrap();

    let script = Script::from_content("function handler(e) { return e; }");
    let result = sandbox.add_handler("", script);
    assert!(result.is_err(), "Empty handler name should be rejected");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("must not be empty"),
        "Error should mention empty name, got: {err}"
    );
}

#[test]
fn remove_handler_rejects_empty_name() {
    let proto = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto.load_runtime().unwrap();

    let result = sandbox.remove_handler("");
    assert!(result.is_err(), "Empty handler name should be rejected");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("must not be empty"),
        "Error should mention empty name, got: {err}"
    );
}

#[test]
fn handle_event_rejects_empty_name() {
    let handler = Script::from_content("function handler(e) { return e; }");
    let proto = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto.load_runtime().unwrap();
    sandbox.add_handler("handler", handler).unwrap();
    let mut loaded = sandbox.get_loaded_sandbox().unwrap();

    let result = loaded.handle_event("", "{}".to_string(), None);
    assert!(result.is_err(), "Empty handler name should be rejected");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("must not be empty"),
        "Error should mention empty name, got: {err}"
    );
}

// ── Auto-export heuristic tests (issue #39) ──────────────────────────
// The auto-export logic must only detect actual ES export statements,
// not the word "export" inside string literals, comments, or identifiers.

#[test]
fn handler_with_export_in_string_literal() {
    // "export" appears inside a string — auto-export should still fire
    let handler = Script::from_content(
        r#"
        function handler(event) {
            const xml = '<config mode="export">value</config>';
            return { result: xml };
        }
        "#,
    );

    let proto = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto.load_runtime().unwrap();
    sandbox.add_handler("handler", handler).unwrap();
    let mut loaded = sandbox.get_loaded_sandbox().unwrap();

    let res = loaded
        .handle_event("handler", "{}".to_string(), None)
        .unwrap();
    assert_eq!(
        res,
        r#"{"result":"<config mode=\"export\">value</config>"}"#
    );
}

#[test]
fn handler_with_export_in_comment() {
    // "export" appears in a comment — auto-export should still fire
    let handler = Script::from_content(
        r#"
        function handler(event) {
            // TODO: export this data to CSV
            return { result: 42 };
        }
        "#,
    );

    let proto = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto.load_runtime().unwrap();
    sandbox.add_handler("handler", handler).unwrap();
    let mut loaded = sandbox.get_loaded_sandbox().unwrap();

    let res = loaded
        .handle_event("handler", "{}".to_string(), None)
        .unwrap();
    assert_eq!(res, r#"{"result":42}"#);
}

#[test]
fn handler_with_export_in_identifier() {
    // "export" is part of an identifier — auto-export should still fire
    let handler = Script::from_content(
        r#"
        function handler(event) {
            const exportPath = "/tmp/out.csv";
            return { result: exportPath };
        }
        "#,
    );

    let proto = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto.load_runtime().unwrap();
    sandbox.add_handler("handler", handler).unwrap();
    let mut loaded = sandbox.get_loaded_sandbox().unwrap();

    let res = loaded
        .handle_event("handler", "{}".to_string(), None)
        .unwrap();
    assert_eq!(res, r#"{"result":"/tmp/out.csv"}"#);
}

#[test]
fn handler_with_explicit_export_is_not_doubled() {
    // Script already has an export statement — auto-export should be skipped
    let handler = Script::from_content(
        r#"
        function handler(event) {
            return { result: "explicit" };
        }
        export { handler };
        "#,
    );

    let proto = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto.load_runtime().unwrap();
    sandbox.add_handler("handler", handler).unwrap();
    let mut loaded = sandbox.get_loaded_sandbox().unwrap();

    let res = loaded
        .handle_event("handler", "{}".to_string(), None)
        .unwrap();
    assert_eq!(res, r#"{"result":"explicit"}"#);
}

#[test]
fn handler_with_export_default_function() {
    // `export function` — auto-export should be skipped
    let handler = Script::from_content(
        r#"
        export function handler(event) {
            return { result: "inline-export" };
        }
        "#,
    );

    let proto = SandboxBuilder::new().build().unwrap();
    let mut sandbox = proto.load_runtime().unwrap();
    sandbox.add_handler("handler", handler).unwrap();
    let mut loaded = sandbox.get_loaded_sandbox().unwrap();

    let res = loaded
        .handle_event("handler", "{}".to_string(), None)
        .unwrap();
    assert_eq!(res, r#"{"result":"inline-export"}"#);
}
