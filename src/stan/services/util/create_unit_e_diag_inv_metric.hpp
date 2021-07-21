#ifndef STAN_SERVICES_UTIL_CREATE_UNIT_E_DIAG_INV_METRIC_HPP
#define STAN_SERVICES_UTIL_CREATE_UNIT_E_DIAG_INV_METRIC_HPP

#include <stan/io/dump.hpp>
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
inline stan::io::dump create_unit_e_diag_inv_metric(size_t num_params) {
  std::string dims("),.Dim=c(" + std::to_string(num_params) + "))");
  Eigen::IOFormat RFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ",",
                       "", "", "inv_metric <- structure(c(", dims);
  std::stringstream txt;
  txt << Eigen::VectorXd::Ones(num_params).format(RFmt);
  return stan::io::dump(txt);
}
}  // namespace util
}  // namespace services
}  // namespace stan

#endif
