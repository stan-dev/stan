#ifndef STAN_SERVICES_UTIL_READ_DIAG_MASS_MATRIX_HPP
#define STAN_SERVICES_UTIL_READ_DIAG_MASS_MATRIX_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/io/var_context.hpp>
#include <Eigen/Dense>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace stan {
  namespace services {
    namespace util {

      /**
       * Extract diagonal values for a mass matrix from a var_context object.
       *
       * @param[in] init_mass_matrix a var_context with initial values
       * @param[in] num_params expected number of diagonal elements
       * @param[in,out] logger Logger for messages
       * @throws std::domain_error if the mass matrix is invalid
       * @return mass_matrix vector of diagonal values
       */
      Eigen::VectorXd
      read_diag_mass_matrix(stan::io::var_context& init_mass_matrix,
                            size_t num_params,
                            callbacks::logger& logger) {
        Eigen::VectorXd inv_mass_matrix(num_params);
        try {
          init_mass_matrix.validate_dims("read diag mass matrix", "mass_matrix",
                                         "vector_d",
                                         init_mass_matrix.to_vec(num_params));
          std::vector<double> diag_vals =
            init_mass_matrix.vals_r("mass_matrix");
          for (size_t i=0; i < num_params; i++) {
            inv_mass_matrix(i) = diag_vals[i];
          }
        } catch (const std::exception& e) {
          logger.error("Cannot get mass matrix from input file.");
          logger.error("Caught exception: ");
          logger.error(e.what());
          throw std::domain_error("Initialization failure");
        }
        return inv_mass_matrix;
      }

    }
  }
}

#endif
