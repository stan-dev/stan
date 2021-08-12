#ifndef STAN_SERVICES_UTIL_CREATE_UNIT_E_DENSE_INV_METRIC_HPP
#define STAN_SERVICES_UTIL_CREATE_UNIT_E_DENSE_INV_METRIC_HPP

#include <stan/io/dump.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
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
inline stan::io::dump create_unit_e_dense_inv_metric(size_t num_params) {
  auto num_params_str = std::to_string(num_params);
  std::string dims("),.Dim=c(" + num_params_str + ", " + num_params_str + "))");
  Eigen::IOFormat RFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ",",
                       "", "", "inv_metric <- structure(c(", dims);
  std::stringstream txt;
  txt << Eigen::MatrixXd::Identity(num_params, num_params).format(RFmt);
  return stan::io::dump(txt);
}
}  // namespace util
}  // namespace services
}  // namespace stan

#endif
