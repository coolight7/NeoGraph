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

#include <cstdint>
#include <string>
#include <vector>

namespace neograph {

/**
 * @brief 消息标记位掩码，用于标识 ChatMessage 的特殊属性。
 *
 * 使用位掩码保持内存紧凑且易于组合判断。
 * 这些标记不会被序列化到 OpenAI API 请求中（由 messages_to_json 过滤）。
 */
enum class MessageFlag : uint32_t {
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
    Outdated = 1 << 4,
};

inline MessageFlag operator|(MessageFlag a, MessageFlag b) {
    return static_cast<MessageFlag>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline MessageFlag operator&(MessageFlag a, MessageFlag b) {
    return static_cast<MessageFlag>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline MessageFlag& operator|=(MessageFlag& a, MessageFlag b) {
    a = a | b;
    return a;
}
inline bool hasFlag(MessageFlag flags, MessageFlag test) {
    return (static_cast<uint32_t>(flags & test)) != 0;
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
 * multi-modal content via image_urls for vision-capable models.
 */
struct ChatMessage {
    std::string           role;        ///< Message role: "user", "assistant", "tool", or "system".
    std::string           content;     ///< Text content of the message.
    std::vector<ToolCall> tool_calls;  ///< Tool calls made by the assistant (if any).
    std::string tool_call_id;          ///< ID of the tool call being responded to (role == "tool").
    std::string tool_name;             ///< Name of the tool being called.
    std::vector<std::string> image_urls;  ///< Base64 data URLs or HTTP URLs for vision support.

    /// [@coolight] 用于支持 修改、重新生成 消息历史
    std::vector<std::string> history_contents;
    std::string              summaryContent;

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

    /// Token usage statistics for the completion.
    struct Usage {
        int prompt_tokens     = 0;  ///< Number of tokens in the prompt.
        int completion_tokens = 0;  ///< Number of tokens in the completion.
        int total_tokens      = 0;  ///< Total tokens used (prompt + completion).
    } usage;
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
    if (!msg.history_contents.empty()) j["history_contents"] = msg.history_contents;
    if (msg.flags != MessageFlag::None) j["flags"] = static_cast<uint32_t>(msg.flags);
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
    if (j.contains("history_contents") && j["history_contents"].is_array()) {
        msg.history_contents = j["history_contents"].get<std::vector<std::string>>();
    }
    if (j.contains("flags")) {
        msg.flags = static_cast<MessageFlag>(j["flags"].get<uint32_t>());
    }
    if (j.contains("extra")) {
        msg.extra = j["extra"];
    }
}

// --- JSON serialization helpers ---

/**
 * @brief Convert a vector of ChatMessages to OpenAI-compatible JSON format.
 *
 * Handles tool call messages, tool result messages, and multi-modal
 * messages (text + images in OpenAI Vision format).
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
        } else if (!msg.image_urls.empty()) {
            // Multi-modal: text + images (OpenAI Vision format)
            json parts = json::array();
            if (!msg.content.empty()) {
                parts.push_back({{"type", "text"}, {"text", msg.content}});
            }
            for (auto& url : msg.image_urls) {
                parts.push_back({{"type", "image_url"}, {"image_url", {{"url", url}}}});
            }
            j["content"] = parts;
        } else {
            j["content"] = msg.content;
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
        arr.push_back({{"type", "function"},
                       {"function",
                        {{"name", tool.name},
                         {"description", tool.description},
                         {"parameters", tool.parameters}}}});
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