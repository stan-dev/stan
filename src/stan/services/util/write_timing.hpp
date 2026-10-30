#ifndef STAN_SERVICES_UTIL_WRITE_TIMING_HPP
#define STAN_SERVICES_UTIL_WRITE_TIMING_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <sstream>
#include <string>

namespace stan {
namespace services {
namespace util {

/**
 * Internal method to write timing information to a writer or logger.
 *
 * @param[in] delta_t time in seconds
 * @param[in] label label for the timing info
 * @param[in] writer output stream or logger
 */
template <typename F>
void write_timing(double delta_t, const std::string& label, F& writer) {
  std::string title(" Elapsed Time: ");
  writer("");

  std::stringstream ss;
  ss << title << delta_t << " seconds (" << label << ")";
  writer(ss.str());

  writer("");
}

/**
 * Write timing information to both writer and logger.
 *
 * @param[in] delta_t time in seconds
 * @param[in] label label for the timing info
 * @param[in,out] writer output stream
 * @param[in,out] logger messages are written through the logger
 */
inline void write_timing(double delta_t, const std::string& label,
                         callbacks::writer& writer, callbacks::logger& logger) {
  write_timing(delta_t, label, writer);
  auto logger_info = [&logger](const std::string& msg) { logger.info(msg); };
  write_timing(delta_t, label, logger_info);
}

}  // namespace util
}  // namespace services
}  // namespace stan

#endif
