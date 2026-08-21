/**
 * @file types.h
 * @brief Foundation types for NeoGraph: messages, tool calls, and LLM completions.
 *
 * Defines the core data structures shared across all NeoGraph modules,
 * including ChatMessage, ToolCall, ChatCompletion, and their JSON
 * serialization helpers (ADL-based, nlohmann/json compatible).
 */
#pragma once

#include <neograph/define.h>
#include <neograph/json.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace neograph {

/**
 * @brief 消息标记位掩码，用于标识 ChatMessage 的特殊属性。
 *
 * 使用位掩码保持内存紧凑且易于组合判断。
 * 这些标记不会被序列化到 OpenAI API 请求中（由 messages_to_json 过滤）。
 */
enum class MessageFlag : uint64_t {
    None = 0,
    /// 由 Agent 运行时自动插入的提示/修复消息（非用户/LLM 原生产出）
    AutoInserted = 1 << 0,
    /// share_store toolcall 的结果已被缩略/截断处理
    ShareStoreTruncated = 1 << 1,
    /// 消息内容已被卸载到 share_store（content 中仅保留 id 引用）
    ContentOffloaded = 1 << 2,
    /// 消息已被总结压缩（原始内容在 summaryContent 中）
    Summarized = 1 << 3,
    /// 消息因过期被标记为无效（如 outdated toolcall）
    Outdated  = 1 << 4,
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
inline bool hasFlag(MessageFlag flags, MessageFlag test) {
    return (static_cast<uint64_t>(flags & test)) != 0;
}

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
 * @brief A message in the conversation history.
 *
 * Supports all standard roles (user, assistant, tool, system) and
 * multi-modal content via image_urls / audio_urls / video_urls for
 * vision / audio / video capable models.
 *
 * 每个 media 条目可以是:
 *   - HTTP(S) URL, 或
 *   - data URL (RFC 2397): "data:<mime>;base64,<payload>"
 *     (音频必须使用 data URL, 各 API 的 input_audio 只接受 base64 数据;
 *      图片/视频两者皆可, 由 provider 按目标 API 能力转换)
 */
struct ChatMessage {
    std::string           role;        ///< Message role: "user", "assistant", "tool", or "system".
    std::string           content;     ///< Text content of the message.
    std::vector<ToolCall> tool_calls;  ///< Tool calls made by the assistant (if any).
    std::string tool_call_id;          ///< ID of the tool call being responded to (role == "tool").
    std::string tool_name;             ///< Name of the tool being called.
    std::vector<std::string> image_urls;  ///< Base64 data URLs or HTTP URLs for vision support.
    /// 音频附件: data URL 或 HTTP URL (data URL 才能转换为各 API 的 input_audio)
    std::vector<std::string> audio_urls;
    /// 视频附件: data URL 或 HTTP URL
    std::vector<std::string> video_urls;

    /// [@coolight] 用于支持 修改、重新生成 消息历史
    std::vector<std::string> history_contents;
    std::string              reasoning_content;

    /// 消息标记位掩码（不序列化到 LLM API 请求）
    MessageFlag flags = MessageFlag::None;
    /// 扩展元数据，用于存储标记相关的附加信息（如 share_store id 等）
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
    json        parameters;   ///< JSON Schema object describing the tool's parameters.
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
        int prompt_tokens     = 0;  ///< Number of tokens in the prompt.
        int completion_tokens = 0;  ///< Number of tokens in the completion.
        int total_tokens      = 0;  ///< Total tokens used (prompt + completion).
        int reasoning_tokens  = 0;  ///< Number of reasoning tokens (if reported).
    } usage;
};

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
    /// call site — otherwise the total silently under-reports for those.
    void add(const ChatCompletion::Usage& u) {
        const long long total = u.total_tokens != 0
                                    ? u.total_tokens
                                    : static_cast<long long>(u.prompt_tokens) + u.completion_tokens;

        prompt_.fetch_add(u.prompt_tokens, std::memory_order_relaxed);
        completion_.fetch_add(u.completion_tokens, std::memory_order_relaxed);
        total_.fetch_add(total, std::memory_order_relaxed);
        reasoning_.fetch_add(u.reasoning_tokens, std::memory_order_relaxed);
    }

    /// Read the running total. Not a consistent snapshot across the three
    /// counters under concurrent writes — read it when the run is done.
    ChatCompletion::Usage snapshot() const {
        ChatCompletion::Usage u;
        u.prompt_tokens     = static_cast<int>(prompt_.load(std::memory_order_relaxed));
        u.completion_tokens = static_cast<int>(completion_.load(std::memory_order_relaxed));
        u.total_tokens      = static_cast<int>(total_.load(std::memory_order_relaxed));
        u.reasoning_tokens  = static_cast<int>(reasoning_.load(std::memory_order_relaxed));
        return u;
    }

private:
    std::atomic<long long> prompt_{0};
    std::atomic<long long> completion_{0};
    std::atomic<long long> total_{0};
    std::atomic<long long> reasoning_{0};
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
    tc.id        = j.value("id", "");
    tc.name      = j.value("name", "");
    tc.arguments = j.value("arguments", "");
}

/// @brief Serialize a ChatMessage to JSON.
/// @param[out] j Target JSON object.
/// @param[in] msg ChatMessage to serialize.
inline void to_json(json& j, const ChatMessage& msg) {
    j["role"]    = msg.role;
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
    if (!msg.tool_name.empty()) j["tool_name"] = msg.tool_name;
    if (!msg.image_urls.empty()) j["image_urls"] = msg.image_urls;
    if (!msg.audio_urls.empty()) j["audio_urls"] = msg.audio_urls;
    if (!msg.video_urls.empty()) j["video_urls"] = msg.video_urls;
    if (!msg.history_contents.empty()) j["history_contents"] = msg.history_contents;
    if (!msg.reasoning_content.empty()) j["reasoning_content"] = msg.reasoning_content;
    if (msg.flags != MessageFlag::None) j["flags"] = static_cast<uint64_t>(msg.flags);
    if (!msg.extra.empty()) j["extra"] = msg.extra;
}

inline void to_json(json& j, const std::vector<ChatMessage>& msgs) {
    j = json::array();
    for (const auto& msg : msgs) {
        json msgJson;
        to_json(msgJson, msg);
        j.push_back(msgJson);
    }
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
    if (j.contains("reasoning_content") && j["reasoning_content"].is_string()) {
        msg.reasoning_content = j["reasoning_content"].get<std::string>();
    }
    if (j.contains("flags")) {
        msg.flags = static_cast<MessageFlag>(j["flags"].get<uint64_t>());
    }
    if (j.contains("extra")) {
        msg.extra = j["extra"];
    }
}

inline void to_json(json& j, const neograph::ChatStreamChunk& e) {
    j = json{{"type", e.type}, {"data", e.data}};
}

inline void from_json(const json& j, neograph::ChatStreamChunk& e) {
    e.type = j.value<int>("type", 0);
    e.data = j.value<std::string>("data", "");
}

// --- JSON serialization helpers ---

/// @brief 解析 data URL (RFC 2397): "data:[<mediatype>][;base64],<data>"
///
/// 仅支持 base64 编码的 data URL; 非 data URL 或无法解析时返回 nullopt。
/// @return 成功返回 {media_type, base64 payload}; media_type 可能为空字符串
///         (data:;base64,xxx 形式), 由调用方决定如何处理。
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
    std::string_view header  = url.substr(0, comma);
    std::string      payload{url.substr(comma + 1)};

    std::string_view mime = header;
    bool             isBase64 = false;
    auto             semi     = header.find(';');
    if (semi != std::string_view::npos) {
        mime            = header.substr(0, semi);
        auto params     = header.substr(semi + 1);
        // 逐段检查参数, 兼容 "data:image/png;base64,xxx" 与
        // "data:image/png;name=a.png;base64,xxx" 等形式
        while (!params.empty()) {
            auto next  = params.find(';');
            auto param = params.substr(0, next);
            if (param == "base64") {
                isBase64 = true;
            }
            params = (next == std::string_view::npos) ? std::string_view{}
                                                      : params.substr(next + 1);
        }
    }
    if (!isBase64) {
        // 非 base64 (percent-encoded) data URL 暂不支持
        return std::nullopt;
    }
    return std::make_pair(std::string(mime), std::move(payload));
}

/// @brief 从 media type 推导各 API 需要的 audio/video format 标识。
///
/// 归一化常见别名并去除参数 (如 ";codecs=..."):
///   audio/wav|x-wav|wave → wav; audio/mpeg|audio/mp3|audio/mpga → mp3;
///   audio/ogg → ogg; audio/aac → aac; audio/flac → flac; audio/webm → webm;
///   video/mp4 → mp4; video/mpeg → mpeg; video/quicktime → mov;
///   video/webm → webm; video/x-msvideo → avi; video/x-matroska → mkv;
///   video/mp2t → mpegts; video/vnd.apple.mpegurl → m3u8
/// 其他类型返回 "/" 之后的子类型原文 (小写), 无 "/" 返回原文小写。
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
    auto    slash   = m.find('/');
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

/// @brief 判断字符串是否为 HTTP(S) URL (供 media 附件转换时区分 data URL 与远程 URL)
inline bool is_http_url(std::string_view url) {
    return url.starts_with("http://") || url.starts_with("https://");
}

/**
 * @brief Convert a vector of ChatMessages to OpenAI-compatible JSON format.
 *
 * Handles tool call messages, tool result messages, and multi-modal
 * messages (text + images/audio/video) in OpenAI multimodal format:
 *   - image: {"type":"image_url","image_url":{"url":<url>}}
 *   - audio: {"type":"input_audio","input_audio":{"data":<base64>,"format":<fmt>}}
 *            (data URL 转换为 data+format; HTTP URL 无法转为 base64, 以
 *             {"url":...} 扩展字段透传, 供兼容网关使用)
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
            j["content"]      = msg.content;
            j["tool_call_id"] = msg.tool_call_id;
        } else if (!msg.tool_calls.empty()) {
            j["content"] = msg.content.empty() ? json(nullptr) : json(msg.content);
            json tc_arr  = json::array();
            for (const auto& tc : msg.tool_calls) {
                tc_arr.push_back({{"id", tc.id},
                                  {"type", "function"},
                                  {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
            }
            j["tool_calls"] = tc_arr;
        } else if (!msg.image_urls.empty() || !msg.audio_urls.empty() || !msg.video_urls.empty()) {
            // Multi-modal: text + images/audio/video (OpenAI multimodal format)
            json parts = json::array();
            if (!msg.content.empty()) {
                parts.push_back({{"type", "text"}, {"text", msg.content}});
            }
            for (const auto& url : msg.image_urls) {
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
                    // HTTP URL 或无法解析的 data URL: 以 url 扩展字段透传
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

        if (!msg.reasoning_content.empty()) {
            j["reasoning_content"] = msg.reasoning_content;
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
        // parameters 缺失/null 时兜底为空对象 schema (部分严格网关如 SCNet
        // 会因 "parameters": null 返回 400 "Format Error")
        json params = tool.parameters.is_object() ? tool.parameters : json::object();
        arr.push_back({{"type", "function"},
                       {"function",
                        {{"name", tool.name},
                         {"description", tool.description},
                         {"parameters", std::move(params)}}}});
    }
    return arr;
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
    auto        m = choice.at("message");
    msg.role      = m.value("role", "assistant");
    msg.content =
        (m.contains("content") && !m["content"].is_null()) ? m["content"].get<std::string>() : "";
    msg.reasoning_content = (m.contains("reasoning_content") && !m["reasoning_content"].is_null())
                                ? m["reasoning_content"].get<std::string>()
                                : "";
    if (msg.reasoning_content.empty() && m.contains("thinking") && !m["thinking"].is_null()) {
        msg.reasoning_content = m["thinking"].get<std::string>();
    }
    // Vercel AI Gateway / 部分网关把推理内容放在 message.reasoning 字段
    if (msg.reasoning_content.empty() && m.contains("reasoning") && !m["reasoning"].is_null()) {
        msg.reasoning_content = m["reasoning"].get<std::string>();
    }

    if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
        for (const auto& tc : m["tool_calls"]) {
            ToolCall call;
            call.id        = tc.value("id", "");
            auto fn        = tc.at("function");
            call.name      = fn.value("name", "");
            call.arguments = fn.value("arguments", "");
            msg.tool_calls.push_back(std::move(call));
        }
    }

    return msg;
}

}  // namespace neograph