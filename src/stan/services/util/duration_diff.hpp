#ifndef STAN_SERVICES_UTIL_DURATION_DIFF
#define STAN_SERVICES_UTIL_DURATION_DIFF

#include <chrono>
#include <cstddef>

namespace stan {
namespace services {
namespace util {

/**
 * Convert time points into a count in seconds.
 * @tparam Scale The scale of the output. For example, if Scale is 1000, the
 * output will be in units of seconds with millisecond precision.
 * @tparam T A type that whose output from a `operator-` is accepted by duration
 * cast. This is normally a
 * `std::chrono::time_point<std::chrono::high_resolution_clock>` from
 *  `std::chrono::high_resolution_clock::now()`.
 * @param start Starting time point
 * @param end Ending time point
 * @return A duration in units of seconds with millisecond precision
 */
template <std::size_t Scale = 1000, typename T>
inline double duration_diff(const T& start, const T& end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count()
         / static_cast<double>(Scale);
}
}  // namespace util
}  // namespace services
}  // namespace stan

#endif
