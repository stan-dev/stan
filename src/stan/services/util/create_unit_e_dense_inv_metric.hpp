#ifndef STAN_SERVICES_UTIL_CREATE_UNIT_E_DENSE_INV_METRIC_HPP
#define STAN_SERVICES_UTIL_CREATE_UNIT_E_DENSE_INV_METRIC_HPP

#include <stan/io/array_var_context.hpp>
#include <sstream>

namespace stan {
namespace services {
namespace util {

/**
 * Create a stan::dump object which contains vector "metric"
 * of specified size where all elements are ones.
 *
 * @param[in] num_params expected number of dense elements
 * @return var_context
 */
inline auto create_unit_e_dense_inv_metric(size_t num_params) {
  std::vector<std::string> names{"inv_metric"};
  std::vector<double> vals(num_params * num_params, 0.0);
  for (size_t i = 0; i < num_params; ++i) {
    vals[i * num_params + i] = 1.0;
  }
  std::vector<std::vector<size_t>> dimss{{num_params, num_params}};

  return stan::io::array_var_context(names, vals, dimss);
}
}  // namespace util
}  // namespace services
}  // namespace stan

#endif
