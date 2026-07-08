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