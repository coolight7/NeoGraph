/**
 * @file types.h
 * @brief Foundation types for NeoGraph: messages, tool calls, and LLM completions.
 *
 * Defines the core data structures shared across all NeoGraph modules,
 * including ChatMessage, ToolCall, ChatCompletion, and their JSON
 * serialization helpers (ADL-based, nlohmann/json compatible).
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <neograph/define.h>
#include <neograph/json.h>

namespace neograph {

/**
 * @brief Represents a single tool invocation requested by the LLM.
 *
 * When an LLM response contains tool calls, each call is represented
 * as a ToolCall with a unique ID, the tool name, and its arguments
 * serialized as a JSON string.
 */
struct ToolCall {
    std::string id;         ///< Unique identifier for this tool call.
    std::string name;       ///< Name of the tool to invoke.
    std::string arguments;  ///< JSON-encoded string of tool arguments.
};

/**
 * @brief Bit mask describing host-visible properties of a ChatMessage.
 *
 * Kept as a mask so several properties combine in one field and the memory
 * stays small. NeoGraph itself never interprets these bits: they travel with
 * the message through serialization for the host (an agent runtime) to use,
 * and message-request builders leave them out of the LLM API payload.
 */
enum class MessageFlag : uint64_t {
    None = 0,
    /// Inserted by the host runtime rather than produced by the user or the model.
    AutoInserted = 1 << 0,
    /// A tool result that the host shortened before storing it.
    ShareStoreTruncated = 1 << 1,
    /// Content moved to the host's store; `content` keeps only a reference.
    ContentOffloaded = 1 << 2,
    /// Replaced by a summary; the replaced text is kept by the host.
    Summarized = 1 << 3,
    /// No longer valid, e.g. a tool call that was superseded.
    Outdated = 1 << 4,
    /// The tool call behind this message was interrupted (HITL).
    Interrupt = 1 << 5,
};

inline MessageFlag operator|(MessageFlag a, MessageFlag b) {
    return static_cast<MessageFlag>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline MessageFlag operator&(MessageFlag a, MessageFlag b) {
    return static_cast<MessageFlag>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline MessageFlag& operator|=(MessageFlag& a, MessageFlag b) {
    a = a | b;
    return a;
}
/// @brief Whether [test] is set in [flags].
inline bool hasFlag(MessageFlag flags, MessageFlag test) {
    return (static_cast<uint64_t>(flags & test)) != 0;
}

/**
 * @brief A message in the conversation history.
 *
 * Supports all standard roles (user, assistant, tool, system) and
 * multi-modal content via image_urls for vision-capable models.
 */
struct ChatMessage {
    std::string role;                    ///< Message role: "user", "assistant", "tool", or "system".
    std::string content;                 ///< Text content of the message.
    std::vector<ToolCall> tool_calls;    ///< Tool calls made by the assistant (if any).
    std::string tool_call_id;            ///< ID of the tool call being responded to (role == "tool").
    std::string tool_name;               ///< Name of the tool being called.
    /// Typed terminal status emitted by the shared ToolExecutionController.
    std::string tool_status;
    bool tool_retryable = false;
    bool tool_effect_uncertain = false;
    std::vector<std::string> image_urls; ///< Base64 data URLs or HTTP URLs for vision support.
    /// Base64 data URLs or HTTP URLs for audio attachments (`input_audio` on
    /// the wire: only a data URL can be converted, a plain HTTP URL cannot).
    std::vector<std::string> audio_urls;
    /// Base64 data URLs or HTTP URLs for video attachments.
    std::vector<std::string> video_urls;
    /// Provider-returned reasoning text. Keep separate from user-visible content.
    std::string reasoning_content;
    /// Opaque provider-native continuation blocks replayed only on a compatible route.
    json reasoning_details = json::array();

    /// Earlier versions of this message, oldest first. A host that rewrites or
    /// regenerates history keeps the replaced content here so the UI can offer
    /// it back without a second store.
    std::vector<std::string> history_contents;

    /// Bit mask of host-used message properties (see MessageFlag).
    MessageFlag flags = MessageFlag::None;
    /// Host-owned side data for this message (share-store ids, timings, ...).
    /// NeoGraph copies it through serialization; the keys are the host's.
    json extra;
};

/**
 * @brief Tool definition metadata sent to the LLM.
 *
 * Describes a callable tool with its name, description, and parameter
 * schema (JSON Schema object) so the LLM can decide when and how to call it.
 */
struct ChatTool {
    std::string name;         ///< Tool name (must be unique within a session).
    std::string description;  ///< Human-readable description of what the tool does.
    json parameters;          ///< JSON Schema object describing the tool's parameters.
};

/**
 * @brief LLM completion response including the message and token usage.
 */
struct ChatCompletion {
    ChatMessage message;  ///< The response message from the LLM.

    /// Normalized reason the provider stopped: `end_turn`, `max_tokens`,
    /// `stop_sequence`, `tool_use`, `content_filter`, `refusal`, or `unknown`.
    /// Adding this field changes the C++ ABI; recompile consumers with this release.
    std::string stop_reason = "unknown";

    /// Token usage statistics for the completion.
    struct Usage {
        int prompt_tokens = 0;      ///< Number of tokens in the prompt.
        int completion_tokens = 0;  ///< Number of tokens in the completion.
        int total_tokens = 0;       ///< Total tokens used (prompt + completion).
        int cached_prompt_tokens = 0; ///< Prompt-token subset served from cache.
        int reasoning_tokens = 0;   ///< Completion-token subset spent on reasoning.
    } usage;
};

/**
 * @brief One streamed piece of a completion, tagged with what it carries.
 *
 * `StreamCallback` only reports text, so a host that renders thinking
 * separately (or persists it as a message field) cannot tell reasoning tokens
 * from answer tokens. Providers that can distinguish the two report through
 * `FormatDataStreamCallback` / `Provider::invoke_format_data` and use these
 * tags; the plain string callback stays available unchanged.
 */
class ChatStreamChunk {
public:
    inline static constexpr int TYPE_CONTENT  = 1;
    inline static constexpr int TYPE_THINKING = 1 << 2;
    inline static constexpr int TYPE_UNKNOWN  = 1 << 30;

    int         type = TYPE_CONTENT;
    std::string data;
};

/**
 * @brief Running total of the token usage of a graph run (issue #88).
 *
 * One of these rides on `RunContext` for the length of a run, exactly as the
 * cancel token does, and is surfaced as `RunResult::usage` when the run ends.
 * It is shared — by the parent run and every subgraph beneath it, and by every
 * branch of a fan-out — so the counters are atomic.
 *
 * **Where it gets fed.** At the node that *receives* a completion, never at the
 * provider that produced it. `RateLimitedProvider` wraps another provider and
 * delegates to it, so a provider-layer counter would count the same completion
 * once per layer. A completion reaches a node exactly once, whatever it went
 * through on the way.
 */
class NEOGRAPH_API UsageAccumulator {
public:
    /// Fold one completion's usage into the running total.
    ///
    /// Providers that report only `prompt_tokens` and `completion_tokens` and
    /// leave `total_tokens` at zero are normalized here rather than at every
    /// call site. Invalid negative values are ignored, and a total smaller
    /// than the component sum is promoted to that sum so a provider cannot
    /// bypass a model-token ceiling by under-reporting usage.
    void add(const ChatCompletion::Usage& u) {
        std::lock_guard lock(mutex_);
        add_locked(u);
    }

    /// Reserve tokens before dispatching a bounded provider request.
    bool try_reserve(long long tokens, long long ceiling) {
        if (tokens <= 0 || ceiling <= 0) return false;
        std::lock_guard lock(mutex_);
        const auto actual    = total_.load(std::memory_order_relaxed);
        const auto committed = saturating_sum(actual, reserved_);
        if (committed > ceiling || tokens > ceiling - committed) return false;
        reserved_ = saturating_sum(reserved_, tokens);
        return true;
    }

    /// Release a reservation when a provider request is not dispatched.
    ///
    /// The separate reservation count makes an over-sized release harmless:
    /// it can consume only reservations, never usage already reported by a provider.
    void release_reservation(long long tokens) {
        if (tokens <= 0) return;
        std::lock_guard lock(mutex_);
        const auto released = std::min(tokens, reserved_);
        reserved_ -= released;
    }

    /// Replace a reservation with the provider's actual usage.
    void settle_reservation(long long reserved,
                            const ChatCompletion::Usage& u) {
        std::lock_guard lock(mutex_);
        const long long requested = std::max(0LL, reserved);
        const long long held      = std::min(requested, reserved_);
        if (held != 0) reserved_ -= held;
        add_locked(u);
    }

    /// Read the running total. Not a consistent snapshot across the three
    /// counters under concurrent writes — read it when the run is done.
    ChatCompletion::Usage snapshot() const noexcept {
        ChatCompletion::Usage u;
        u.prompt_tokens     = public_counter(prompt_.load(std::memory_order_relaxed));
        u.completion_tokens = public_counter(completion_.load(std::memory_order_relaxed));
        u.total_tokens      = public_counter(total_.load(std::memory_order_relaxed));
        return u;
    }

    /// Read the wide running total for hard budget comparisons.
    long long total_tokens_wide() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

private:
    static long long nonnegative(int value) noexcept {
        return value > 0 ? static_cast<long long>(value) : 0;
    }

    static long long normalized_total(const ChatCompletion::Usage& u,
                                      long long                    prompt,
                                      long long                    completion) noexcept {
        const long long components =
            prompt > std::numeric_limits<long long>::max() - completion
                ? std::numeric_limits<long long>::max()
                : prompt + completion;
        const long long reported = u.total_tokens > 0
                                        ? static_cast<long long>(u.total_tokens)
                                        : 0;
        return std::max(reported, components);
    }

    static long long saturating_sum(long long current, long long delta) noexcept {
        if (current < 0) current = 0;
        if (delta <= 0) return current;
        const auto available = std::numeric_limits<long long>::max() - current;
        return delta > available ? std::numeric_limits<long long>::max()
                                 : current + delta;
    }

    void add_locked(const ChatCompletion::Usage& u) {
        const long long prompt     = nonnegative(u.prompt_tokens);
        const long long completion = nonnegative(u.completion_tokens);
        const long long total      = normalized_total(u, prompt, completion);
        add_counters_locked(prompt, completion, total);
    }

    void add_counters_locked(long long prompt,
                             long long completion,
                             long long total) {
        prompt_.store(saturating_sum(prompt_.load(std::memory_order_relaxed), prompt),
                      std::memory_order_relaxed);
        completion_.store(
            saturating_sum(completion_.load(std::memory_order_relaxed), completion),
            std::memory_order_relaxed);
        total_.store(saturating_sum(total_.load(std::memory_order_relaxed), total),
                     std::memory_order_relaxed);
    }

    static int public_counter(long long value) noexcept {
        if (value <= 0) return 0;
        const auto maximum = static_cast<long long>(std::numeric_limits<int>::max());
        return static_cast<int>(std::min(value, maximum));
    }

    mutable std::mutex     mutex_;
    std::atomic<long long> prompt_{0};
    std::atomic<long long> completion_{0};
    std::atomic<long long> total_{0};
    long long              reserved_ = 0;
};

// --- ADL serialization: ChatMessage/ToolCall <-> json ---
// These live in the same namespace as the types for ADL lookup.

/// @brief Serialize a ToolCall to JSON.
/// @param[out] j Target JSON object.
/// @param[in] tc ToolCall to serialize.
inline void to_json(json& j, const ToolCall& tc) {
    j = json{{"id", tc.id}, {"name", tc.name}, {"arguments", tc.arguments}};
}

/// @brief Deserialize a ToolCall from JSON.
/// @param[in] j Source JSON object.
/// @param[out] tc Target ToolCall.
inline void from_json(const json& j, ToolCall& tc) {
    tc.id = j.value("id", "");
    tc.name = j.value("name", "");
    tc.arguments = j.value("arguments", "");
}

/// @brief Serialize a ChatMessage to JSON.
/// @param[out] j Target JSON object.
/// @param[in] msg ChatMessage to serialize.
inline void to_json(json& j, const ChatMessage& msg) {
    j["role"] = msg.role;
    j["content"] = msg.content;
    if (!msg.tool_calls.empty()) {
        j["tool_calls"] = json::array();
        for (const auto& tc : msg.tool_calls) {
            json tc_j;
            to_json(tc_j, tc);
            j["tool_calls"].push_back(tc_j);
        }
    }
    if (!msg.tool_call_id.empty()) j["tool_call_id"] = msg.tool_call_id;
    if (!msg.tool_name.empty())    j["tool_name"] = msg.tool_name;
    if (!msg.tool_status.empty())  j["tool_status"] = msg.tool_status;
    if (msg.tool_retryable)        j["tool_retryable"] = true;
    if (msg.tool_effect_uncertain) j["tool_effect_uncertain"] = true;
    if (!msg.image_urls.empty())   j["image_urls"] = msg.image_urls;
    if (!msg.audio_urls.empty())   j["audio_urls"] = msg.audio_urls;
    if (!msg.video_urls.empty())   j["video_urls"] = msg.video_urls;
    if (!msg.history_contents.empty()) j["history_contents"] = msg.history_contents;
    if (!msg.reasoning_content.empty()) j["reasoning_content"] = msg.reasoning_content;
    if (!msg.reasoning_details.empty()) j["reasoning_details"] = msg.reasoning_details;
    if (msg.flags != MessageFlag::None) j["flags"] = static_cast<uint64_t>(msg.flags);
    if (!msg.extra.empty())        j["extra"] = msg.extra;
}

/// @brief Deserialize a ChatMessage from JSON.
/// @param[in] j Source JSON object.
/// @param[out] msg Target ChatMessage.
inline void from_json(const json& j, ChatMessage& msg) {
    msg.role    = j.value("role", "");
    msg.content = j.value("content", "");
    if (j.contains("tool_calls") && j["tool_calls"].is_array()) {
        for (const auto& tc_j : j["tool_calls"]) {
            ToolCall tc;
            from_json(tc_j, tc);
            msg.tool_calls.push_back(tc);
        }
    }
    msg.tool_call_id = j.value("tool_call_id", "");
    msg.tool_name    = j.value("tool_name", "");
    msg.tool_status  = j.value("tool_status", "");
    msg.tool_retryable = j.value("tool_retryable", false);
    msg.tool_effect_uncertain = j.value("tool_effect_uncertain", false);
    if (j.contains("image_urls") && j["image_urls"].is_array()) {
        msg.image_urls = j["image_urls"].get<std::vector<std::string>>();
    }
    if (j.contains("audio_urls") && j["audio_urls"].is_array()) {
        msg.audio_urls = j["audio_urls"].get<std::vector<std::string>>();
    }
    if (j.contains("video_urls") && j["video_urls"].is_array()) {
        msg.video_urls = j["video_urls"].get<std::vector<std::string>>();
    }
    if (j.contains("history_contents") && j["history_contents"].is_array()) {
        msg.history_contents = j["history_contents"].get<std::vector<std::string>>();
    }
    // `reasoning` is the key an older revision wrote; accept it so records
    // stored before this field was renamed still load.
    if (j.contains("reasoning_content") && j["reasoning_content"].is_string()) {
        msg.reasoning_content = j["reasoning_content"].get<std::string>();
    } else {
        msg.reasoning_content = j.value("reasoning", "");
    }
    if (j.contains("flags")) {
        msg.flags = static_cast<MessageFlag>(j["flags"].get<uint64_t>());
    }
    if (j.contains("extra")) {
        msg.extra = j["extra"];
    }
    if (j.contains("reasoning_details") && !j["reasoning_details"].is_array()) {
        throw std::invalid_argument("ChatMessage reasoning_details must be an array");
    }
    msg.reasoning_details = j.contains("reasoning_details")
        ? j["reasoning_details"]
        : json::array();
}

// --- JSON serialization helpers ---

/// @brief Parse a base64 data URL (RFC 2397): "data:[<mediatype>][;base64],<data>".
///
/// Only base64-encoded data URLs are supported; anything else (a plain URL,
/// a percent-encoded payload, or a malformed header) returns std::nullopt.
/// @return `{media_type, base64 payload}` on success, where the media type may
///         be empty (the `data:;base64,...` form); the caller decides how to
///         treat an empty type.
inline std::optional<std::pair<std::string, std::string>> parse_data_url(std::string_view url) {
    constexpr std::string_view kPrefix = "data:";
    if (!url.starts_with(kPrefix)) {
        return std::nullopt;
    }
    url.remove_prefix(kPrefix.size());
    auto comma = url.find(',');
    if (comma == std::string_view::npos) {
        return std::nullopt;
    }
    std::string_view header = url.substr(0, comma);
    std::string      payload{url.substr(comma + 1)};

    std::string_view mime     = header;
    bool             is_base64 = false;
    auto             semi      = header.find(';');
    if (semi != std::string_view::npos) {
        mime       = header.substr(0, semi);
        auto params = header.substr(semi + 1);
        // Walk the parameter list so both "data:image/png;base64,..." and
        // "data:image/png;name=a.png;base64,..." are accepted.
        while (!params.empty()) {
            auto next  = params.find(';');
            auto param = params.substr(0, next);
            if (param == "base64") {
                is_base64 = true;
            }
            params = (next == std::string_view::npos) ? std::string_view{}
                                                      : params.substr(next + 1);
        }
    }
    if (!is_base64) {
        // Percent-encoded data URLs are not supported yet.
        return std::nullopt;
    }
    return std::make_pair(std::string(mime), std::move(payload));
}

/// @brief Derive the audio/video `format` identifier the APIs expect.
///
/// Common aliases are normalized and parameters dropped (e.g. ";codecs=..."),
/// so callers can pass the media type straight from a data URL:
///   audio/wav|x-wav|wave -> wav; audio/mpeg|audio/mp3|audio/mpga -> mp3;
///   audio/ogg -> ogg; audio/aac -> aac; audio/flac -> flac; audio/webm -> webm;
///   video/mp4 -> mp4; video/mpeg -> mpeg; video/quicktime -> mov;
///   video/webm -> webm; video/x-msvideo -> avi; video/x-matroska -> mkv;
///   video/mp2t -> mpegts; video/vnd.apple.mpegurl -> m3u8
/// Anything else returns the lower-cased subtype (the text after "/"), or the
/// whole lower-cased media type when there is no "/".
inline std::string media_format_from_mime(std::string_view mime) {
    std::string m(mime);
    for (auto& c : m) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    auto semi = m.find(';');
    if (semi != std::string::npos) {
        m.resize(semi);
    }
    auto        slash   = m.find('/');
    std::string type    = (slash == std::string::npos) ? m : m.substr(0, slash);
    std::string subtype = (slash == std::string::npos) ? m : m.substr(slash + 1);

    if (subtype == "x-wav" || subtype == "wave") {
        return "wav";
    }
    if (subtype == "mpeg" || subtype == "mpga") {
        return (type == "audio") ? "mp3" : "mpeg";
    }
    if (subtype == "quicktime") {
        return "mov";
    }
    if (subtype == "x-msvideo") {
        return "avi";
    }
    if (subtype == "x-matroska") {
        return "mkv";
    }
    if (subtype == "mp2t") {
        return "mpegts";
    }
    if (subtype == "vnd.apple.mpegurl") {
        return "m3u8";
    }
    return subtype;
}

/**
 * @brief Convert a vector of ChatMessages to OpenAI-compatible JSON format.
 *
 * Handles tool call messages, tool result messages, and multi-modal
 * messages (text plus images/audio/video) in OpenAI multimodal format:
 *   - image: {"type":"image_url","image_url":{"url":<url>}}
 *   - audio: {"type":"input_audio","input_audio":{"data":<base64>,"format":<fmt>}}
 *            (a data URL becomes data+format; an HTTP URL cannot be turned
 *             into base64, so it is passed through as {"url":...} for
 *             gateways that accept it)
 *   - video: {"type":"video_url","video_url":{"url":<url>}}
 *
 * @param messages Vector of ChatMessage objects to convert.
 * @return JSON array in OpenAI messages format.
 */
inline json messages_to_json(const std::vector<ChatMessage>& messages) {
    json arr = json::array();
    for (const auto& msg : messages) {
        json j;
        j["role"] = msg.role;

        if (msg.role == "tool") {
            j["content"] = msg.content;
            j["tool_call_id"] = msg.tool_call_id;
        } else if (!msg.tool_calls.empty()) {
            j["content"] = msg.content.empty() ? json(nullptr) : json(msg.content);
            json tc_arr = json::array();
            for (const auto& tc : msg.tool_calls) {
                tc_arr.push_back({
                    {"id", tc.id},
                    {"type", "function"},
                    {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                });
            }
            j["tool_calls"] = tc_arr;
        } else if (!msg.image_urls.empty() || !msg.audio_urls.empty() ||
                   !msg.video_urls.empty()) {
            // Multi-modal: text + images/audio/video (OpenAI multimodal format)
            json parts = json::array();
            if (!msg.content.empty()) {
                parts.push_back({{"type", "text"}, {"text", msg.content}});
            }
            for (auto& url : msg.image_urls) {
                parts.push_back({{"type", "image_url"}, {"image_url", {{"url", url}}}});
            }
            for (const auto& url : msg.audio_urls) {
                if (auto parsed = parse_data_url(url)) {
                    parts.push_back({
                        {"type", "input_audio"},
                        {"input_audio",
                         {{"data", parsed->second},
                          {"format", media_format_from_mime(parsed->first)}}},
                    });
                } else {
                    // An HTTP URL (or an unparsable data URL) is passed through
                    // as a url field; only a data URL can become base64.
                    parts.push_back({
                        {"type", "input_audio"},
                        {"input_audio", {{"url", url}}},
                    });
                }
            }
            for (const auto& url : msg.video_urls) {
                parts.push_back({{"type", "video_url"}, {"video_url", {{"url", url}}}});
            }
            j["content"] = parts;
        } else {
            j["content"] = msg.content;
        }

        if (msg.role == "assistant") {
            if (!msg.reasoning_details.empty()) {
                if (!msg.reasoning_details.is_array()) {
                    throw std::invalid_argument(
                        "ChatMessage reasoning_details must be an array");
                }
                j["reasoning_details"] = msg.reasoning_details;
            } else if (!msg.reasoning_content.empty()) {
                j["reasoning_content"] = msg.reasoning_content;
            }
        }

        arr.push_back(j);
    }
    return arr;
}

/**
 * @brief Convert a vector of ChatTools to OpenAI-compatible JSON format.
 *
 * @param tools Vector of ChatTool objects to convert.
 * @return JSON array in OpenAI tool definition format.
 */
inline json tools_to_json(const std::vector<ChatTool>& tools) {
    json arr = json::array();
    for (const auto& tool : tools) {
        // A tool without a parameter schema would serialize as
        // "parameters": null, which strict gateways reject with
        // 400 "Format Error"; fall back to an empty object schema.
        json params = tool.parameters.is_object() ? tool.parameters : json::object();
        arr.push_back({
            {"type", "function"},
            {"function", {
                {"name", tool.name},
                {"description", tool.description},
                {"parameters", std::move(params)}
            }}
        });
    }
    return arr;
}

/// @brief Serialize a message list as a JSON array of ChatMessage objects.
///
/// This is the lossless form (it keeps the host fields too), unlike
/// messages_to_json(), which builds the OpenAI request shape.
inline void to_json(json& j, const std::vector<ChatMessage>& msgs) {
    j = json::array();
    for (const auto& msg : msgs) {
        json msg_json;
        to_json(msg_json, msg);
        j.push_back(std::move(msg_json));
    }
}

/// @brief Serialize one streamed chunk (tag + payload).
inline void to_json(json& j, const neograph::ChatStreamChunk& e) {
    j = json{{"type", e.type}, {"data", e.data}};
}

/// @brief Deserialize one streamed chunk.
inline void from_json(const json& j, neograph::ChatStreamChunk& e) {
    e.type = j.value<int>("type", 0);
    e.data = j.value<std::string>("data", "");
}

/**
 * @brief Parse an OpenAI API response choice into a ChatMessage.
 *
 * Extracts the message content, role, and any tool calls from
 * the `choices[n]` object of an OpenAI completion response.
 *
 * @param choice A single choice object from the OpenAI response (must contain "message").
 * @return Parsed ChatMessage with role, content, and tool_calls populated.
 * @throws json::exception If required fields are missing.
 */
inline ChatMessage parse_response_message(const json& choice) {
    ChatMessage msg;
    auto m = choice.at("message");
    msg.role = m.value("role", "assistant");
    msg.content = (m.contains("content") && !m["content"].is_null())
                  ? m["content"].get<std::string>() : "";
    if (m.contains("reasoning_content") && !m["reasoning_content"].is_null()) {
        msg.reasoning_content = m["reasoning_content"].get<std::string>();
    }
    // Providers disagree on the field name for the trace and it may sit next
    // to "content" instead of inside it: DeepSeek-style gateways answer with
    // "reasoning_content", several OpenAI-compatible servers use "thinking",
    // and Vercel AI Gateway uses "reasoning".
    if (msg.reasoning_content.empty() && m.contains("thinking") && !m["thinking"].is_null()) {
        msg.reasoning_content = m["thinking"].get<std::string>();
    }
    if (msg.reasoning_content.empty() && m.contains("reasoning") && !m["reasoning"].is_null()) {
        msg.reasoning_content = m["reasoning"].get<std::string>();
    }
    if (m.contains("reasoning_details") && m["reasoning_details"].is_array()) {
        msg.reasoning_details = m["reasoning_details"];
    }

    if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
        for (const auto& tc : m["tool_calls"]) {
            ToolCall call;
            call.id = tc.value("id", "");
            auto fn = tc.at("function");
            call.name = fn.value("name", "");
            call.arguments = fn.value("arguments", "");
            msg.tool_calls.push_back(std::move(call));
        }
    }

    return msg;
}

} // namespace neograph
