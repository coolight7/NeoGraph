/**
 * @file define.h
 * @brief Asio source selection for downstream projects.
 *
 * NeoGraph normally builds against the vendored standalone asio under
 * `deps/asio/include`. A downstream project that already uses Boost.Asio
 * can point the `asio` CMake target at its Boost installation instead
 * (see the `asio` target in the top-level CMakeLists.txt). In that mode
 * `NEOGRAPH_USE_BOOST_ASIO` is defined and this header:
 *
 *   - exposes `boost::asio` under the `asio` name, matching the way such
 *     projects include asio headers (`<asio/...>` mapped to
 *     `<boost/asio/...>` by their include path), and
 *   - provides the error types under names that exist in both modes.
 *     Standalone asio uses `std::error_code` / `std::system_error` and
 *     never declares a `boost::asio::error_code`, so the `asio::error_code`
 *     spelling used by the vendored headers cannot compile against Boost.
 */
#pragma once

#include <type_traits>

static_assert(__cplusplus >= 201703, "Requires __cplusplus >= 201703 (C++17 or later)");
static_assert(__cpp_lib_is_invocable >= 201703,
              "Requires __cpp_lib_is_invocable >= 201703 (C++17 or later)");

#include <asio/awaitable.hpp>

#ifdef NEOGRAPH_USE_BOOST_ASIO
namespace asio                   = ::boost::asio;
using neograph_asio_system_error = ::boost::system::system_error;
using neograph_asio_error_code   = ::boost::system::error_code;
#else
using neograph_asio_system_error = ::asio::system_error;
using neograph_asio_error_code   = ::asio::error_code;
#endif
