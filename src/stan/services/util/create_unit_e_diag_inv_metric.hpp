#ifndef STAN_SERVICES_UTIL_CREATE_UNIT_E_DIAG_INV_METRIC_HPP
#define STAN_SERVICES_UTIL_CREATE_UNIT_E_DIAG_INV_METRIC_HPP

#include <stan/io/array_var_context.hpp>
#include <sstream>

namespace stan {
namespace services {
namespace util {

/**
 * Create a stan::dump object which contains vector "metric"
 * of specified size where all elements are ones.
 *
 * @param[in] num_params expected number of diagonal elements
 * @return var_context
 */
inline auto create_unit_e_diag_inv_metric(size_t num_params) {
  std::vector<std::string> names{"inv_metric"};
  std::vector<double> vals(num_params, 1.0);
  std::vector<std::vector<size_t>> dimss{{num_params}};

  return stan::io::array_var_context(names, vals, dimss);
}
}  // namespace util
}  // namespace services
}  // namespace stan

#endif
