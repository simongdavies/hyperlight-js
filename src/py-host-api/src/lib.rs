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

//! Python host bindings for `hyperlight-js`.
//!
//! This crate mirrors the `js-host-api` (Node/NAPI) binding, exposing the
//! staged sandbox lifecycle to Python:
//!
//! `SandboxBuilder` → `build()` → `ProtoJSSandbox` → `load_runtime()` →
//! `JSSandbox` → `get_loaded_sandbox()` → `LoadedJSSandbox`.
//!
//! Host functions are **synchronous** Python callables. Unlike the Node bridge
//! (which drives a Promise via a `ThreadsafeFunction` + oneshot channel), the
//! guest VCPU blocks the calling thread while a host function runs, so a plain
//! synchronous call under the GIL is sufficient. All blocking VM calls release
//! the GIL via `Python::detach`, and host-function / host-print callbacks
//! re-acquire it via `Python::attach`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use hl_core::{
    CpuTimeMonitor, ExecutionStats, FnReturn, HyperlightError as HlError,
    InterruptHandle as HlInterruptHandle, JSSandbox as HlJSSandbox,
    LoadedJSSandbox as HlLoadedJSSandbox, ProtoJSSandbox as HlProtoJSSandbox,
    SandboxBuilder as HlSandboxBuilder, Script, Snapshot as HlSnapshot, WallClockMonitor,
};
use hyperlight_js_common::PLACEHOLDER_BIN;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyString, PyStringMethods, PyTuple};
use serde_json::Value as JsonValue;

// ── Limits (kept in sync with the Node binding) ──────────────────────

/// Minimum allowed timeout value in milliseconds.
const MIN_TIMEOUT_MS: u32 = 1;
/// Maximum allowed timeout value in milliseconds (1 hour).
const MAX_TIMEOUT_MS: u32 = 3_600_000;
/// Maximum allowed length for a module or namespace identifier.
const MAX_MODULE_IDENTIFIER_LEN: usize = 256;

// ── Exception hierarchy ──────────────────────────────────────────────

create_exception!(
    hyperlight_js,
    HyperlightError,
    PyException,
    "Base class for all errors raised by the hyperlight-js host binding."
);
create_exception!(
    hyperlight_js,
    PoisonedError,
    HyperlightError,
    "The sandbox is in a poisoned (inconsistent) state — restore or unload it."
);
create_exception!(
    hyperlight_js,
    CancelledError,
    HyperlightError,
    "Execution was cancelled by the host (monitor timeout or manual kill())."
);
create_exception!(
    hyperlight_js,
    GuestAbortError,
    HyperlightError,
    "The guest aborted (trap, panic, or fatal error in guest code)."
);
create_exception!(
    hyperlight_js,
    InvalidArgError,
    HyperlightError,
    "Invalid arguments (bad types, empty names, out-of-range sizes/timeouts)."
);
create_exception!(
    hyperlight_js,
    ConsumedError,
    HyperlightError,
    "The object has already been consumed — each lifecycle transition is one-shot."
);
create_exception!(
    hyperlight_js,
    InternalError,
    HyperlightError,
    "Internal / unexpected failure (lock poison, serialization bug, etc.)."
);

/// Build an `ERR_INVALID_ARG` Python error.
fn invalid_arg(msg: impl Into<String>) -> PyErr {
    InvalidArgError::new_err(format!("[ERR_INVALID_ARG] {}", msg.into()))
}

/// Build an `ERR_CONSUMED` Python error.
fn consumed(type_name: &str) -> PyErr {
    ConsumedError::new_err(format!(
        "[ERR_CONSUMED] {type_name} has already been consumed — each instance can only be used once"
    ))
}

/// Build an `ERR_INTERNAL` Python error.
fn internal(msg: impl Into<String>) -> PyErr {
    InternalError::new_err(format!("[ERR_INTERNAL] {}", msg.into()))
}

/// Error for a poisoned Rust-level `Mutex` (a bug, distinct from a poisoned sandbox).
fn lock_err() -> PyErr {
    internal("internal lock poisoned — this is a bug")
}

/// Map a [`HlError`] to the matching typed Python exception, mirroring the
/// Node binding's `ErrorCode` mapping.
fn map_hl_err(err: HlError) -> PyErr {
    let msg = err.to_string();
    match &err {
        HlError::PoisonedSandbox => PoisonedError::new_err(format!("[ERR_POISONED] {msg}")),
        HlError::ExecutionCanceledByHost() => {
            CancelledError::new_err(format!("[ERR_CANCELLED] {msg}"))
        }
        HlError::JsonConversionFailure(_) => {
            InvalidArgError::new_err(format!("[ERR_INVALID_ARG] {msg}"))
        }
        HlError::GuestAborted(_, _) => GuestAbortError::new_err(format!("[ERR_GUEST_ABORT] {msg}")),
        _ => InternalError::new_err(format!("[ERR_INTERNAL] {msg}")),
    }
}

// ── Module identifier validation (mirrors js_sandbox.rs / the Node layer) ──

fn validate_module_identifier(value: &str, label: &str) -> PyResult<()> {
    if value.is_empty() || value.trim().is_empty() {
        return Err(invalid_arg(format!("{label} must not be empty")));
    }
    if value.contains(':') {
        return Err(invalid_arg(format!("{label} must not contain ':'")));
    }
    if value.chars().any(|c| c.is_control()) {
        return Err(invalid_arg(format!(
            "{label} must not contain control characters"
        )));
    }
    if value.len() > MAX_MODULE_IDENTIFIER_LEN {
        return Err(invalid_arg(format!(
            "{label} must not exceed {MAX_MODULE_IDENTIFIER_LEN} characters"
        )));
    }
    Ok(())
}

fn validate_namespace_not_reserved(namespace: &str) -> PyResult<()> {
    if namespace == "host" {
        return Err(invalid_arg(format!(
            "Module namespace '{namespace}' is reserved"
        )));
    }
    Ok(())
}

// ── JSON <-> Python conversion ───────────────────────────────────────

/// Convert a Python value into a `serde_json::Value`.
///
/// `bytes` / `bytearray` are intentionally rejected here — binary only crosses
/// the boundary as a top-level host-function return value, never inside an
/// event payload or nested structure.
fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if obj.is_none() {
        return Ok(JsonValue::Null);
    }
    // bool must be checked before int (Python `bool` is a subclass of `int`).
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(JsonValue::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(JsonValue::Number(i.into()));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(JsonValue::Number(u.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return json_number_from_f64(f);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(JsonValue::String(s.to_cow()?.into_owned()));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(py_to_json(&item)?);
        }
        return Ok(JsonValue::Array(arr));
    }
    if let Ok(tup) = obj.cast::<PyTuple>() {
        let mut arr = Vec::with_capacity(tup.len());
        for item in tup.iter() {
            arr.push(py_to_json(&item)?);
        }
        return Ok(JsonValue::Array(arr));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key = k
                .cast::<PyString>()
                .map_err(|_| invalid_arg("dict keys must be strings to convert to JSON"))?
                .to_cow()?
                .into_owned();
            map.insert(key, py_to_json(&v)?);
        }
        return Ok(JsonValue::Object(map));
    }
    Err(invalid_arg(
        "unsupported value for JSON conversion (expected None/bool/int/float/str/list/tuple/dict)",
    ))
}

/// Build a JSON number from an `f64`, rejecting non-finite values.
fn json_number_from_f64(v: f64) -> PyResult<JsonValue> {
    serde_json::Number::from_f64(v)
        .map(JsonValue::Number)
        .ok_or_else(|| invalid_arg("non-finite float (NaN/Infinity) cannot be represented in JSON"))
}

/// If `map` is a `{"__bin__": N}` placeholder and `blobs` are available, return
/// the referenced blob.
fn bin_placeholder<'a>(
    map: &serde_json::Map<String, JsonValue>,
    blobs: Option<&'a [Vec<u8>]>,
) -> Option<&'a [u8]> {
    let blobs = blobs?;
    if map.len() == 1
        && let Some(JsonValue::Number(n)) = map.get(PLACEHOLDER_BIN)
        && let Some(idx) = n.as_u64()
    {
        return blobs.get(idx as usize).map(Vec::as_slice);
    }
    None
}

/// Convert a `serde_json::Value` into a Python object.
///
/// When `blobs` is `Some`, `{"__bin__": N}` placeholder objects are replaced
/// with `bytes` taken from `blobs[N]`.
fn json_to_py<'py>(
    py: Python<'py>,
    value: &JsonValue,
    blobs: Option<&[Vec<u8>]>,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        JsonValue::Null => Ok(py.None().into_bound(py)),
        JsonValue::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any()),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.into_any())
            } else {
                let f = n.as_f64().unwrap_or(0.0);
                Ok(f.into_pyobject(py)?.into_any())
            }
        }
        JsonValue::String(s) => Ok(PyString::new(py, s).into_any()),
        JsonValue::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item, blobs)?)?;
            }
            Ok(list.into_any())
        }
        JsonValue::Object(map) => {
            if let Some(bytes) = bin_placeholder(map, blobs) {
                return Ok(PyBytes::new(py, bytes).into_any());
            }
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k.as_str(), json_to_py(py, v, blobs)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

/// Convert a host function's Python return value into a [`FnReturn`].
///
/// Top-level `bytes` / `bytearray` become `FnReturn::Binary`; everything else
/// is JSON-serialized into `FnReturn::Json`.
fn py_return_to_fnreturn(result: &Bound<'_, PyAny>) -> PyResult<FnReturn> {
    if let Ok(b) = result.cast::<PyBytes>() {
        return Ok(FnReturn::Binary(b.as_bytes().to_vec()));
    }
    if let Ok(ba) = result.cast::<PyByteArray>() {
        return Ok(FnReturn::Binary(ba.to_vec()));
    }
    if result.is_none() {
        return Ok(FnReturn::Json("null".to_owned()));
    }
    let value = py_to_json(result)?;
    let json = serde_json::to_string(&value)
        .map_err(|e| internal(format!("failed to serialize host return value: {e}")))?;
    Ok(FnReturn::Json(json))
}

// ── SandboxBuilder ───────────────────────────────────────────────────

/// Configures and allocates a sandbox VM.
#[pyclass]
struct SandboxBuilder {
    inner: Arc<Mutex<Option<HlSandboxBuilder>>>,
}

impl SandboxBuilder {
    /// Apply a transformation to the inner builder, or error if consumed.
    fn map_builder(&self, f: impl FnOnce(HlSandboxBuilder) -> HlSandboxBuilder) -> PyResult<()> {
        let mut guard = self.inner.lock().map_err(|_| lock_err())?;
        let builder = guard.take().ok_or_else(|| consumed("SandboxBuilder"))?;
        *guard = Some(f(builder));
        Ok(())
    }
}

#[pymethods]
impl SandboxBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(HlSandboxBuilder::new()))),
        }
    }

    /// Set the guest output buffer size in bytes (must be > 0). Returns `self`.
    fn output_buffer_size(slf: PyRef<'_, Self>, size: usize) -> PyResult<PyRef<'_, Self>> {
        if size == 0 {
            return Err(invalid_arg("output buffer size must be greater than 0"));
        }
        slf.map_builder(|b| b.with_guest_output_buffer_size(size))?;
        Ok(slf)
    }

    /// Set the guest input buffer size in bytes (must be > 0). Returns `self`.
    fn input_buffer_size(slf: PyRef<'_, Self>, size: usize) -> PyResult<PyRef<'_, Self>> {
        if size == 0 {
            return Err(invalid_arg("input buffer size must be greater than 0"));
        }
        slf.map_builder(|b| b.with_guest_input_buffer_size(size))?;
        Ok(slf)
    }

    /// Set the guest scratch (stack) size in bytes (must be > 0). Returns `self`.
    fn scratch_size(slf: PyRef<'_, Self>, size: usize) -> PyResult<PyRef<'_, Self>> {
        if size == 0 {
            return Err(invalid_arg("scratch size must be greater than 0"));
        }
        slf.map_builder(|b| b.with_guest_scratch_size(size))?;
        Ok(slf)
    }

    /// Set the guest heap size in bytes (must be > 0). Returns `self`.
    fn heap_size(slf: PyRef<'_, Self>, size: u64) -> PyResult<PyRef<'_, Self>> {
        if size == 0 {
            return Err(invalid_arg("heap size must be greater than 0"));
        }
        slf.map_builder(|b| b.with_guest_heap_size(size))?;
        Ok(slf)
    }

    /// Set a callback that receives guest `console.log` / `print` output.
    ///
    /// The callback is invoked as `callback(message: str)`. It must not raise;
    /// any exception is swallowed (the guest print is best-effort).
    fn host_print_fn(slf: PyRef<'_, Self>, callback: Py<PyAny>) -> PyResult<PyRef<'_, Self>> {
        slf.map_builder(move |b| {
            let print_fn = move |msg: String| -> i32 {
                Python::attach(|py| match callback.bind(py).call1((msg,)) {
                    Ok(_) => 0,
                    Err(_) => -1,
                })
            };
            b.with_host_print_fn(print_fn.into())
        })?;
        Ok(slf)
    }

    /// Allocate the sandbox VM resources, consuming this builder.
    fn build(&self, py: Python<'_>) -> PyResult<ProtoJSSandbox> {
        let builder = self
            .inner
            .lock()
            .map_err(|_| lock_err())?
            .take()
            .ok_or_else(|| consumed("SandboxBuilder"))?;
        let proto = py.detach(|| builder.build().map_err(map_hl_err))?;
        Ok(ProtoJSSandbox {
            inner: Arc::new(Mutex::new(Some(proto))),
        })
    }
}

// ── ProtoJSSandbox ───────────────────────────────────────────────────

/// A sandbox with VM resources allocated, ready for host-function registration.
#[pyclass]
struct ProtoJSSandbox {
    inner: Arc<Mutex<Option<HlProtoJSSandbox>>>,
}

#[pymethods]
impl ProtoJSSandbox {
    /// Get a builder for registering host functions in a named module.
    ///
    /// Guest code imports these as `import * as name from "host:name"`.
    fn host_module(&self, name: &str) -> PyResult<HostModule> {
        if name.is_empty() {
            return Err(invalid_arg("module name must not be empty"));
        }
        Ok(HostModule {
            inner: self.inner.clone(),
            module_name: format!("host:{name}"),
        })
    }

    /// Convenience for `host_module(module_name).register(function_name, callback)`.
    fn register(
        &self,
        module_name: &str,
        function_name: String,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.host_module(module_name)?
            .register(function_name, callback)
    }

    /// Load the JavaScript runtime, applying all registered host functions.
    /// Consumes this `ProtoJSSandbox`.
    fn load_runtime(&self, py: Python<'_>) -> PyResult<JSSandbox> {
        let proto = self
            .inner
            .lock()
            .map_err(|_| lock_err())?
            .take()
            .ok_or_else(|| consumed("ProtoJSSandbox"))?;
        let js = py.detach(|| proto.load_runtime().map_err(map_hl_err))?;
        Ok(JSSandbox {
            inner: Arc::new(Mutex::new(Some(js))),
        })
    }
}

// ── HostModule ───────────────────────────────────────────────────────

/// A builder for registering host functions in a named module.
#[pyclass]
struct HostModule {
    inner: Arc<Mutex<Option<HlProtoJSSandbox>>>,
    module_name: String,
}

#[pymethods]
impl HostModule {
    /// Register a synchronous Python callable as a host function.
    ///
    /// Guest arguments arrive spread positionally. `bytes` / `bytearray` guest
    /// arguments are delivered as Python `bytes`; returning `bytes` /
    /// `bytearray` sends binary back to the guest, any other value is sent as
    /// JSON. Registering the same name twice overwrites the prior registration.
    fn register(&self, name: String, callback: Py<PyAny>) -> PyResult<()> {
        if name.is_empty() {
            return Err(invalid_arg("function name must not be empty"));
        }
        let fn_name = name.clone();
        let wrapper = move |args: JsonValue, blobs: Vec<Vec<u8>>| -> hl_core::Result<FnReturn> {
            Python::attach(|py| {
                let outcome: PyResult<FnReturn> = (|| {
                    let py_args: Vec<Bound<'_, PyAny>> = match &args {
                        JsonValue::Array(arr) => arr
                            .iter()
                            .map(|v| json_to_py(py, v, Some(&blobs)))
                            .collect::<PyResult<Vec<_>>>()?,
                        other => vec![json_to_py(py, other, Some(&blobs))?],
                    };
                    let tuple = PyTuple::new(py, py_args)?;
                    let result = callback.bind(py).call1(tuple)?;
                    py_return_to_fnreturn(&result)
                })();
                outcome
                    .map_err(|e| HlError::Error(format!("host function '{fn_name}' raised: {e}")))
            })
        };
        let mut guard = self.inner.lock().map_err(|_| lock_err())?;
        let proto = guard.as_mut().ok_or_else(|| consumed("ProtoJSSandbox"))?;
        proto
            .host_module(&self.module_name)
            .register_js(name, wrapper);
        Ok(())
    }
}

// ── JSSandbox ────────────────────────────────────────────────────────

/// A sandbox with the JS runtime loaded, ready for handler/module registration.
#[pyclass]
struct JSSandbox {
    inner: Arc<Mutex<Option<HlJSSandbox>>>,
}

impl JSSandbox {
    fn with_mut<R>(&self, f: impl FnOnce(&mut HlJSSandbox) -> PyResult<R>) -> PyResult<R> {
        let mut guard = self.inner.lock().map_err(|_| lock_err())?;
        let sandbox = guard.as_mut().ok_or_else(|| consumed("JSSandbox"))?;
        f(sandbox)
    }
}

#[pymethods]
impl JSSandbox {
    /// Register a named handler. `script` must define a function named `handler`.
    fn add_handler(&self, name: String, script: String) -> PyResult<()> {
        if name.is_empty() {
            return Err(invalid_arg("handler name must not be empty"));
        }
        self.with_mut(|s| {
            s.add_handler(name, Script::from_content(script))
                .map_err(map_hl_err)
        })
    }

    /// Remove a previously registered handler by name.
    fn remove_handler(&self, name: &str) -> PyResult<()> {
        if name.is_empty() {
            return Err(invalid_arg("handler name must not be empty"));
        }
        self.with_mut(|s| s.remove_handler(name).map_err(map_hl_err))
    }

    /// Remove all registered handlers.
    fn clear_handlers(&self) -> PyResult<()> {
        self.with_mut(|s| {
            s.clear_handlers();
            Ok(())
        })
    }

    /// Register a user ES module, importable as `<namespace>:<name>`
    /// (default namespace `"user"`).
    #[pyo3(signature = (name, source, namespace=None))]
    fn add_module(&self, name: String, source: String, namespace: Option<String>) -> PyResult<()> {
        validate_module_identifier(&name, "Module name")?;
        if let Some(ns) = &namespace {
            validate_module_identifier(ns, "Module namespace")?;
            validate_namespace_not_reserved(ns)?;
        }
        self.with_mut(|s| {
            match namespace {
                Some(ns) => s.add_module_ns(name, Script::from_content(source), ns),
                None => s.add_module(name, Script::from_content(source)),
            }
            .map_err(map_hl_err)
        })
    }

    /// Remove a previously registered user module (default namespace `"user"`).
    #[pyo3(signature = (name, namespace=None))]
    fn remove_module(&self, name: String, namespace: Option<String>) -> PyResult<()> {
        validate_module_identifier(&name, "Module name")?;
        if let Some(ns) = &namespace {
            validate_module_identifier(ns, "Module namespace")?;
        }
        self.with_mut(|s| {
            match namespace {
                Some(ns) => s.remove_module_ns(&name, &ns),
                None => s.remove_module(&name),
            }
            .map_err(map_hl_err)
        })
    }

    /// Remove all registered user modules.
    fn clear_modules(&self) -> PyResult<()> {
        self.with_mut(|s| {
            s.clear_modules();
            Ok(())
        })
    }

    /// Compile registered handlers into the guest and transition to
    /// `LoadedJSSandbox`. Consumes this `JSSandbox`.
    fn get_loaded_sandbox(&self, py: Python<'_>) -> PyResult<LoadedJSSandbox> {
        let sandbox = self
            .inner
            .lock()
            .map_err(|_| lock_err())?
            .take()
            .ok_or_else(|| consumed("JSSandbox"))?;
        let loaded = py.detach(|| sandbox.get_loaded_sandbox().map_err(map_hl_err))?;
        let interrupt = loaded.interrupt_handle();
        let poisoned_flag = Arc::new(AtomicBool::new(loaded.poisoned()));
        Ok(LoadedJSSandbox {
            inner: Arc::new(Mutex::new(Some(loaded))),
            interrupt,
            poisoned_flag,
            last_call_stats: Arc::new(Mutex::new(None)),
            disposed: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Whether the sandbox is in a poisoned (inconsistent) state.
    #[getter]
    fn poisoned(&self) -> PyResult<bool> {
        self.with_mut(|s| Ok(s.poisoned()))
    }

    /// Eagerly release the sandbox; subsequent calls raise `ConsumedError`.
    fn dispose(&self) -> PyResult<()> {
        let _ = self.inner.lock().map_err(|_| lock_err())?.take();
        Ok(())
    }
}

// ── LoadedJSSandbox ──────────────────────────────────────────────────

/// An execution-ready sandbox. Call `call_handler()` to invoke handlers.
#[pyclass]
struct LoadedJSSandbox {
    inner: Arc<Mutex<Option<HlLoadedJSSandbox>>>,
    /// Stored outside the lock so `kill()` works while a handler is running.
    interrupt: Arc<dyn HlInterruptHandle>,
    /// Lock-free poisoned flag, updated under the call lock after each op.
    poisoned_flag: Arc<AtomicBool>,
    /// Stats from the most recent `call_handler()`, updated under the call lock.
    last_call_stats: Arc<Mutex<Option<CallStats>>>,
    /// Tracks consumption (via `dispose()` / `unload()`) for sync getters.
    disposed: Arc<AtomicBool>,
}

#[pymethods]
impl LoadedJSSandbox {
    /// Invoke a handler with the given event, optionally guarded by timeouts.
    ///
    /// `event` is any JSON-serializable Python value. When a timeout is set,
    /// an execution monitor races the handler with OR semantics — whichever
    /// fires first terminates execution (raising `CancelledError`).
    #[pyo3(signature = (handler_name, event, *, wall_clock_timeout_ms=None, cpu_timeout_ms=None, gc=None))]
    fn call_handler(
        &self,
        py: Python<'_>,
        handler_name: String,
        event: &Bound<'_, PyAny>,
        wall_clock_timeout_ms: Option<u32>,
        cpu_timeout_ms: Option<u32>,
        gc: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if handler_name.is_empty() {
            return Err(invalid_arg("handler name must not be empty"));
        }
        if let Some(w) = wall_clock_timeout_ms
            && !(MIN_TIMEOUT_MS..=MAX_TIMEOUT_MS).contains(&w)
        {
            return Err(invalid_arg(format!(
                "wall_clock_timeout_ms must be between {MIN_TIMEOUT_MS} and {MAX_TIMEOUT_MS}, got {w}"
            )));
        }
        if let Some(c) = cpu_timeout_ms
            && !(MIN_TIMEOUT_MS..=MAX_TIMEOUT_MS).contains(&c)
        {
            return Err(invalid_arg(format!(
                "cpu_timeout_ms must be between {MIN_TIMEOUT_MS} and {MAX_TIMEOUT_MS}, got {c}"
            )));
        }

        let event_json = serde_json::to_string(&py_to_json(event)?)
            .map_err(|e| invalid_arg(format!("failed to serialize event: {e}")))?;

        let inner = self.inner.clone();
        let poisoned_flag = self.poisoned_flag.clone();
        let stats_store = self.last_call_stats.clone();

        let result_json = py.detach(move || -> PyResult<String> {
            let mut guard = inner.lock().map_err(|_| lock_err())?;
            let sandbox = guard.as_mut().ok_or_else(|| consumed("LoadedJSSandbox"))?;

            // The sealed `MonitorSet` trait is not object-safe, so the four
            // (wall, cpu) arms are structurally required — each builds a
            // distinct concrete monitor type.
            let result = match (wall_clock_timeout_ms, cpu_timeout_ms) {
                (None, None) => sandbox
                    .handle_event(handler_name, event_json, gc)
                    .map_err(map_hl_err),
                (Some(wall_ms), Some(cpu_ms)) => {
                    let monitor = (
                        WallClockMonitor::new(Duration::from_millis(wall_ms as u64))
                            .map_err(map_hl_err)?,
                        CpuTimeMonitor::new(Duration::from_millis(cpu_ms as u64))
                            .map_err(map_hl_err)?,
                    );
                    sandbox
                        .handle_event_with_monitor(handler_name, event_json, &monitor, gc)
                        .map_err(map_hl_err)
                }
                (Some(wall_ms), None) => {
                    let monitor = WallClockMonitor::new(Duration::from_millis(wall_ms as u64))
                        .map_err(map_hl_err)?;
                    sandbox
                        .handle_event_with_monitor(handler_name, event_json, &monitor, gc)
                        .map_err(map_hl_err)
                }
                (None, Some(cpu_ms)) => {
                    let monitor = CpuTimeMonitor::new(Duration::from_millis(cpu_ms as u64))
                        .map_err(map_hl_err)?;
                    sandbox
                        .handle_event_with_monitor(handler_name, event_json, &monitor, gc)
                        .map_err(map_hl_err)
                }
            };

            poisoned_flag.store(sandbox.poisoned(), Ordering::Release);
            if let Ok(mut stats) = stats_store.lock() {
                *stats = sandbox.last_call_stats().map(CallStats::from_stats);
            }
            result
        })?;

        let value: JsonValue = serde_json::from_str(&result_json)
            .map_err(|e| internal(format!("failed to parse handler result as JSON: {e}")))?;
        Ok(json_to_py(py, &value, None)?.unbind())
    }

    /// Evaluate arbitrary JavaScript against the sandbox's persistent global
    /// context (REPL semantics) and return the completion value.
    ///
    /// Unlike `call_handler`, this runs `code` as global script code rather
    /// than invoking a named handler. Top-level `var`/`let`/`const`/`function`/
    /// `class` declarations are added to the shared global scope and persist
    /// across subsequent `eval()` and `call_handler()` calls — exactly like a
    /// `quickjs` REPL.
    ///
    /// The completion value is returned as a Python value. A value of
    /// `undefined` — or any value that is not JSON-serializable (e.g. a
    /// function) — is returned as `None` rather than raising.
    ///
    /// When a timeout is set, an execution monitor races the evaluation with
    /// OR semantics — whichever fires first terminates execution (raising
    /// `CancelledError`).
    #[pyo3(signature = (code, *, wall_clock_timeout_ms=None, cpu_timeout_ms=None, gc=None))]
    fn eval(
        &self,
        py: Python<'_>,
        code: String,
        wall_clock_timeout_ms: Option<u32>,
        cpu_timeout_ms: Option<u32>,
        gc: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if code.is_empty() {
            return Err(invalid_arg("eval code must not be empty"));
        }
        if let Some(w) = wall_clock_timeout_ms
            && !(MIN_TIMEOUT_MS..=MAX_TIMEOUT_MS).contains(&w)
        {
            return Err(invalid_arg(format!(
                "wall_clock_timeout_ms must be between {MIN_TIMEOUT_MS} and {MAX_TIMEOUT_MS}, got {w}"
            )));
        }
        if let Some(c) = cpu_timeout_ms
            && !(MIN_TIMEOUT_MS..=MAX_TIMEOUT_MS).contains(&c)
        {
            return Err(invalid_arg(format!(
                "cpu_timeout_ms must be between {MIN_TIMEOUT_MS} and {MAX_TIMEOUT_MS}, got {c}"
            )));
        }

        let inner = self.inner.clone();
        let poisoned_flag = self.poisoned_flag.clone();
        let stats_store = self.last_call_stats.clone();

        let result_json = py.detach(move || -> PyResult<String> {
            let mut guard = inner.lock().map_err(|_| lock_err())?;
            let sandbox = guard.as_mut().ok_or_else(|| consumed("LoadedJSSandbox"))?;

            // The sealed `MonitorSet` trait is not object-safe, so the four
            // (wall, cpu) arms are structurally required — each builds a
            // distinct concrete monitor type.
            let result = match (wall_clock_timeout_ms, cpu_timeout_ms) {
                (None, None) => sandbox.eval(code, gc).map_err(map_hl_err),
                (Some(wall_ms), Some(cpu_ms)) => {
                    let monitor = (
                        WallClockMonitor::new(Duration::from_millis(wall_ms as u64))
                            .map_err(map_hl_err)?,
                        CpuTimeMonitor::new(Duration::from_millis(cpu_ms as u64))
                            .map_err(map_hl_err)?,
                    );
                    sandbox
                        .eval_with_monitor(code, &monitor, gc)
                        .map_err(map_hl_err)
                }
                (Some(wall_ms), None) => {
                    let monitor = WallClockMonitor::new(Duration::from_millis(wall_ms as u64))
                        .map_err(map_hl_err)?;
                    sandbox
                        .eval_with_monitor(code, &monitor, gc)
                        .map_err(map_hl_err)
                }
                (None, Some(cpu_ms)) => {
                    let monitor = CpuTimeMonitor::new(Duration::from_millis(cpu_ms as u64))
                        .map_err(map_hl_err)?;
                    sandbox
                        .eval_with_monitor(code, &monitor, gc)
                        .map_err(map_hl_err)
                }
            };

            poisoned_flag.store(sandbox.poisoned(), Ordering::Release);
            if let Ok(mut stats) = stats_store.lock() {
                *stats = sandbox.last_call_stats().map(CallStats::from_stats);
            }
            result
        })?;

        let value: JsonValue = serde_json::from_str(&result_json)
            .map_err(|e| internal(format!("failed to parse eval result as JSON: {e}")))?;
        Ok(json_to_py(py, &value, None)?.unbind())
    }

    /// Unload handlers and return to the `JSSandbox` state. Consumes `self`.
    fn unload(&self, py: Python<'_>) -> PyResult<JSSandbox> {
        let sandbox = self
            .inner
            .lock()
            .map_err(|_| lock_err())?
            .take()
            .ok_or_else(|| consumed("LoadedJSSandbox"))?;
        self.disposed.store(true, Ordering::Release);
        let js = py.detach(|| sandbox.unload().map_err(map_hl_err))?;
        Ok(JSSandbox {
            inner: Arc::new(Mutex::new(Some(js))),
        })
    }

    /// Capture the current guest state as a snapshot for later `restore()`.
    fn snapshot(&self, py: Python<'_>) -> PyResult<Snapshot> {
        let inner = self.inner.clone();
        let poisoned_flag = self.poisoned_flag.clone();
        let snap = py.detach(move || -> PyResult<Arc<HlSnapshot>> {
            let mut guard = inner.lock().map_err(|_| lock_err())?;
            let sandbox = guard.as_mut().ok_or_else(|| consumed("LoadedJSSandbox"))?;
            let result = sandbox.snapshot().map_err(map_hl_err);
            poisoned_flag.store(sandbox.poisoned(), Ordering::Release);
            result
        })?;
        Ok(Snapshot { inner: snap })
    }

    /// Restore the sandbox to a previously captured snapshot.
    fn restore(&self, py: Python<'_>, snapshot: &Snapshot) -> PyResult<()> {
        let inner = self.inner.clone();
        let poisoned_flag = self.poisoned_flag.clone();
        let snap = snapshot.inner.clone();
        py.detach(move || -> PyResult<()> {
            let mut guard = inner.lock().map_err(|_| lock_err())?;
            let sandbox = guard.as_mut().ok_or_else(|| consumed("LoadedJSSandbox"))?;
            let result = sandbox.restore(snap).map_err(map_hl_err);
            poisoned_flag.store(sandbox.poisoned(), Ordering::Release);
            result
        })
    }

    /// A handle that can `kill()` currently running guest code.
    #[getter]
    fn interrupt_handle(&self) -> PyResult<InterruptHandle> {
        if self.disposed.load(Ordering::Acquire) {
            return Err(consumed("LoadedJSSandbox"));
        }
        Ok(InterruptHandle {
            inner: self.interrupt.clone(),
        })
    }

    /// Whether the sandbox is in a poisoned (inconsistent) state.
    #[getter]
    fn poisoned(&self) -> PyResult<bool> {
        if self.disposed.load(Ordering::Acquire) {
            return Err(consumed("LoadedJSSandbox"));
        }
        Ok(self.poisoned_flag.load(Ordering::Acquire))
    }

    /// Execution statistics from the most recent `call_handler()`, or `None`.
    #[getter]
    fn last_call_stats(&self) -> PyResult<Option<CallStats>> {
        if self.disposed.load(Ordering::Acquire) {
            return Err(consumed("LoadedJSSandbox"));
        }
        Ok(self.last_call_stats.lock().map_err(|_| lock_err())?.clone())
    }

    /// Eagerly release the sandbox; subsequent calls raise `ConsumedError`.
    fn dispose(&self) -> PyResult<()> {
        let _ = self.inner.lock().map_err(|_| lock_err())?.take();
        self.disposed.store(true, Ordering::Release);
        Ok(())
    }
}

// ── Snapshot / InterruptHandle / CallStats ───────────────────────────

/// An opaque guest-state snapshot produced by `LoadedJSSandbox.snapshot()`.
#[pyclass]
struct Snapshot {
    inner: Arc<HlSnapshot>,
}

/// A handle for terminating currently running guest code.
#[pyclass]
struct InterruptHandle {
    inner: Arc<dyn HlInterruptHandle>,
}

#[pymethods]
impl InterruptHandle {
    /// Immediately terminate the currently executing guest code. The sandbox
    /// becomes poisoned; recover with `restore()` or `unload()`.
    fn kill(&self) {
        self.inner.kill();
    }
}

/// Execution statistics from a guest handler call.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
struct CallStats {
    /// Wall-clock (elapsed) time in milliseconds. Always present.
    wall_clock_ms: f64,
    /// CPU time in milliseconds, when the CPU clock handle was available.
    cpu_time_ms: Option<f64>,
    /// Name of the monitor that terminated execution, if any.
    terminated_by: Option<String>,
}

impl CallStats {
    fn from_stats(stats: &ExecutionStats) -> Self {
        Self {
            wall_clock_ms: stats.wall_clock.as_secs_f64() * 1000.0,
            cpu_time_ms: stats.cpu_time.map(|d| d.as_secs_f64() * 1000.0),
            terminated_by: stats.terminated_by.map(|s| s.to_string()),
        }
    }
}

// ── Module ───────────────────────────────────────────────────────────

/// `True` if a supported hypervisor (KVM / WHP / mshv) is available.
#[pyfunction]
fn is_hypervisor_present() -> bool {
    hl_core::is_hypervisor_present()
}

#[pymodule]
fn hyperlight_js(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add_class::<SandboxBuilder>()?;
    m.add_class::<ProtoJSSandbox>()?;
    m.add_class::<HostModule>()?;
    m.add_class::<JSSandbox>()?;
    m.add_class::<LoadedJSSandbox>()?;
    m.add_class::<Snapshot>()?;
    m.add_class::<InterruptHandle>()?;
    m.add_class::<CallStats>()?;
    m.add_function(wrap_pyfunction!(is_hypervisor_present, m)?)?;

    m.add("HyperlightError", py.get_type::<HyperlightError>())?;
    m.add("PoisonedError", py.get_type::<PoisonedError>())?;
    m.add("CancelledError", py.get_type::<CancelledError>())?;
    m.add("GuestAbortError", py.get_type::<GuestAbortError>())?;
    m.add("InvalidArgError", py.get_type::<InvalidArgError>())?;
    m.add("ConsumedError", py.get_type::<ConsumedError>())?;
    m.add("InternalError", py.get_type::<InternalError>())?;

    // Expose the `ERR_*` code on each exception class for callers that prefer
    // string-code matching over `except` by type.
    for (name, code) in [
        ("PoisonedError", "ERR_POISONED"),
        ("CancelledError", "ERR_CANCELLED"),
        ("GuestAbortError", "ERR_GUEST_ABORT"),
        ("InvalidArgError", "ERR_INVALID_ARG"),
        ("ConsumedError", "ERR_CONSUMED"),
        ("InternalError", "ERR_INTERNAL"),
    ] {
        m.getattr(name)?.setattr("code", code)?;
    }

    Ok(())
}
