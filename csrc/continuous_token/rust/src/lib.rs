//! FFI shim binding the real HF `tokenizers` Rust crate (BPE encode) and the
//! `minijinja` + `minijinja-contrib` crates (Jinja2-compatible chat-template
//! rendering) for verl's `ContinuousTokenBuilder.build_initial_tokens` --
//! and *only* that method. `merge_assistant_tokens`/`align_response_metadata`
//! are pure integer/list bookkeeping in the Python source (confirmed by
//! reading `verl/utils/tokenizer/continuous_token.py`) and are ported
//! directly to C++ in `../src/continuous_token.cpp`; they do not touch this
//! crate at all.
//!
//! Scope (see `../README.md` for the full scoping trace): this binds exactly
//! what `SingleTurnAgentLoop.run()` exercises for the validated Qwen3-4B /
//! GSM8K single-turn, text-only workload. `tools` is always `None` on that
//! path, so the `{% if tools %}` chat-template branch (and its `tojson` use)
//! is never reached by this workload; it is left correctly *parseable* (the
//! template is rendered as a whole) but is NOT part of what was validated
//! here -- see the README's honesty section.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use minijinja::value::Value as MjValue;
use minijinja::{context, Environment};
use tokenizers::tokenizer::Tokenizer;

/// Opaque builder handle returned to C++. Holds the loaded HF tokenizer and
/// a minijinja `Environment` configured to match the exact Jinja2 settings
/// `transformers.utils.chat_template_utils._cached_compile_jinja_template`
/// uses (`trim_blocks=True, lstrip_blocks=True`) -- confirmed by reading
/// that function in the pinned `transformers==5.12.1` -- plus the
/// `minijinja_contrib::pycompat` unknown-method callback, which is what
/// makes Python string methods (`.startswith()`, `.split()`, `.lstrip()`,
/// ...) used throughout Qwen's `chat_template.jinja` resolve at all.
pub struct VctBuilder {
    tokenizer: Tokenizer,
    env: Environment<'static>,
}

const TEMPLATE_NAME: &str = "chat_template";

fn set_err(err_out: *mut *mut c_char, msg: impl AsRef<str>) {
    if err_out.is_null() {
        return;
    }
    let c = match CString::new(msg.as_ref()) {
        Ok(c) => c,
        Err(_) => CString::new("error message contained an interior NUL byte").unwrap(),
    };
    unsafe {
        *err_out = c.into_raw();
    }
}

/// Load a tokenizer from a `tokenizer.json` path (the HF-standard format,
/// the same file Qwen3 checkpoints already ship) plus the raw Jinja2
/// chat-template source (verbatim from `tokenizer_config.json`'s
/// `"chat_template"` field -- callers read that file themselves; this shim
/// doesn't re-parse `tokenizer_config.json` to avoid a second JSON schema
/// dependency for one string field).
///
/// On failure returns NULL and (if `err_out` is non-NULL) sets `*err_out` to
/// an owned, NUL-terminated error string the caller must free with
/// `vct_free_string`.
#[no_mangle]
pub extern "C" fn vct_builder_new(
    tokenizer_json_path: *const c_char,
    chat_template_ptr: *const u8,
    chat_template_len: usize,
    err_out: *mut *mut c_char,
) -> *mut VctBuilder {
    if tokenizer_json_path.is_null() || (chat_template_ptr.is_null() && chat_template_len != 0) {
        set_err(err_out, "vct_builder_new: null argument");
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(tokenizer_json_path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_err(err_out, format!("tokenizer_json_path is not valid UTF-8: {e}"));
            return ptr::null_mut();
        }
    };

    let tokenizer = match Tokenizer::from_file(path) {
        Ok(t) => t,
        Err(e) => {
            set_err(err_out, format!("Tokenizer::from_file({path:?}) failed: {e}"));
            return ptr::null_mut();
        }
    };

    let template_bytes = unsafe { std::slice::from_raw_parts(chat_template_ptr, chat_template_len) };
    let template_src = match std::str::from_utf8(template_bytes) {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_err(err_out, format!("chat_template is not valid UTF-8: {e}"));
            return ptr::null_mut();
        }
    };

    // The template source must outlive `env` (minijinja templates borrow
    // their source). Builders are process-lifetime singletons in the real
    // agent loop (one per rollout worker, constructed once) and in this
    // harness, so a one-time intentional leak per builder is the simplest
    // correct option -- documented here rather than reaching for unsafe
    // self-referential-struct tricks for a leak that in practice never
    // accumulates (bounded by the number of builders a process creates,
    // not the number of rollouts).
    let template_src: &'static str = Box::leak(template_src.into_boxed_str());

    let mut env: Environment<'static> = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    if let Err(e) = env.add_template(TEMPLATE_NAME, template_src) {
        set_err(err_out, format!("chat template failed to parse: {e}"));
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(VctBuilder { tokenizer, env }))
}

/// Frees a builder returned by `vct_builder_new`. Safe to call with NULL.
#[no_mangle]
pub extern "C" fn vct_builder_free(builder: *mut VctBuilder) {
    if !builder.is_null() {
        unsafe {
            drop(Box::from_raw(builder));
        }
    }
}

/// Owned array of token ids handed back to C++. Always allocated by Rust and
/// must be released with `vct_free_token_ids` -- never freed with `free()`/
/// `delete[]` on the C++ side, since it crosses the Rust/C allocator
/// boundary (see the C++ RAII wrapper in `../include/verl_continuous_token/continuous_token.h`,
/// which is the only code that's supposed to touch this struct directly).
#[repr(C)]
pub struct VctTokenIds {
    pub data: *mut i64,
    pub len: usize,
}

impl VctTokenIds {
    fn empty() -> Self {
        VctTokenIds { data: ptr::null_mut(), len: 0 }
    }

    fn from_vec(v: Vec<i64>) -> Self {
        let boxed: Box<[i64]> = v.into_boxed_slice();
        let len = boxed.len();
        if len == 0 {
            return VctTokenIds::empty();
        }
        let data = Box::into_raw(boxed) as *mut i64;
        VctTokenIds { data, len }
    }
}

/// Port of `ContinuousTokenBuilder.build_initial_tokens` -> `_render_tokens`
/// -> `verl.utils.tokenizer.chat_template.apply_chat_template` for the
/// single-turn, text-only, no-tools path: render the chat template over
/// `messages` with the given flags, then BPE-encode the rendered text with
/// the real HF tokenizer (`add_special_tokens=false`, matching what
/// `PreTrainedTokenizerBase.apply_chat_template(tokenize=True)` does --
/// confirmed empirically: the template already emits every special token as
/// literal text, and encoding the rendered string with
/// `add_special_tokens=False` reproduced the exact real reference token ids
/// for every generated fixture; see README.md).
///
/// `messages_json` must be a JSON array of objects with at least a `role`
/// and a `content` string field (the only shape `build_initial_tokens`
/// actually receives on the validated single-turn GSM8K path -- no
/// `tool_calls`/`reasoning_content`, since those only appear on assistant
/// messages, and assistant messages are never part of the *initial* prompt).
/// `tools` is always treated as absent/falsy, matching the real call site
/// (`SingleTurnAgentLoop` never passes `tools` to `ct_build_initial_tokens`).
///
/// `enable_thinking_defined`/`enable_thinking_value` mirror whether
/// `chat_template_kwargs` contained an `"enable_thinking"` key in Python --
/// the template checks `enable_thinking is defined`, so an actually-absent
/// kwarg and a present-but-irrelevant one are NOT the same thing to Jinja.
///
/// Returns 0 on success (with `*out` populated; free with
/// `vct_free_token_ids`) or -1 on failure (with `*err_out` populated; free
/// with `vct_free_string`).
#[no_mangle]
pub extern "C" fn vct_build_initial_tokens(
    builder: *mut VctBuilder,
    messages_json_ptr: *const u8,
    messages_json_len: usize,
    add_generation_prompt: bool,
    enable_thinking_defined: bool,
    enable_thinking_value: bool,
    out: *mut VctTokenIds,
    err_out: *mut *mut c_char,
) -> i32 {
    if builder.is_null() || out.is_null() {
        set_err(err_out, "vct_build_initial_tokens: null argument");
        return -1;
    }
    let builder = unsafe { &*builder };

    let messages_bytes = unsafe { std::slice::from_raw_parts(messages_json_ptr, messages_json_len) };
    let messages_json: serde_json::Value = match serde_json::from_slice(messages_bytes) {
        Ok(v) => v,
        Err(e) => {
            set_err(err_out, format!("messages_json is not valid JSON: {e}"));
            unsafe { *out = VctTokenIds::empty() };
            return -1;
        }
    };

    let tmpl = match builder.env.get_template(TEMPLATE_NAME) {
        Ok(t) => t,
        Err(e) => {
            set_err(err_out, format!("internal error: template not found: {e}"));
            unsafe { *out = VctTokenIds::empty() };
            return -1;
        }
    };

    let ctx = if enable_thinking_defined {
        context! {
            messages => MjValue::from_serialize(&messages_json),
            add_generation_prompt => add_generation_prompt,
            tools => MjValue::UNDEFINED,
            enable_thinking => enable_thinking_value,
        }
    } else {
        context! {
            messages => MjValue::from_serialize(&messages_json),
            add_generation_prompt => add_generation_prompt,
            tools => MjValue::UNDEFINED,
        }
    };

    let rendered = match tmpl.render(ctx) {
        Ok(s) => s,
        Err(e) => {
            set_err(err_out, format!("chat template render failed: {e}"));
            unsafe { *out = VctTokenIds::empty() };
            return -1;
        }
    };

    let encoding = match builder.tokenizer.encode(rendered, false) {
        Ok(e) => e,
        Err(e) => {
            set_err(err_out, format!("tokenizer encode failed: {e}"));
            unsafe { *out = VctTokenIds::empty() };
            return -1;
        }
    };

    let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    unsafe { *out = VctTokenIds::from_vec(ids) };
    0
}

/// Frees a `VctTokenIds` returned by `vct_build_initial_tokens`. Safe to call
/// on an already-empty (NULL data / zero len) value.
#[no_mangle]
pub extern "C" fn vct_free_token_ids(ids: VctTokenIds) {
    if !ids.data.is_null() && ids.len > 0 {
        unsafe {
            let slice_ptr = std::ptr::slice_from_raw_parts_mut(ids.data, ids.len);
            drop(Box::from_raw(slice_ptr));
        }
    }
}

/// Frees an error string set by any `*_err_out` parameter above.
#[no_mangle]
pub extern "C" fn vct_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}
