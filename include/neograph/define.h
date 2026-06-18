#pragma once

#include <asio/awaitable.hpp>

#ifdef NEOGRAPH_USE_BOODT_ASIO
namespace asio                   = ::boost::asio;
using neograph_asio_system_error = ::boost::system::system_error;
using neograph_asio_error_code   = ::boost::system::error_code;
#else
using neograph_asio_system_error = ::asio::system_error;
using neograph_asio_error_code   = ::asio::error_code;
#endif