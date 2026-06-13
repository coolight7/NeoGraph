/**
 * @file tool.cpp
 * @brief Out-of-line definition of AsyncTool::execute (Sem 4.2).
 *
 * Kept in its own TU so the run_sync template instantiation cost is
 * paid here once, not at every include of <neograph/tool.h>.
 */
#include <neograph/async/run_sync.h>
#include <neograph/tool.h>

namespace neograph {

asio::awaitable<std::string> Tool::real_execute_async(const json& arguments) {
    co_return execute(arguments);
}

std::string AsyncTool::execute(const json& arguments) {
    return neograph::async::run_sync(execute_async(arguments));
}
asio::awaitable<std::string> AsyncTool::real_execute_async(const json& arguments) {
    co_return co_await execute_async(arguments);
}

}  // namespace neograph
