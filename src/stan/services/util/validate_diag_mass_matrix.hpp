#ifndef STAN_SERVICES_UTIL_VALIDATE_DIAG_MASS_MATRIX_HPP
#define STAN_SERVICES_UTIL_VALIDATE_DIAG_MASS_MATRIX_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
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
       * @param[in,out] error_writer message writer
       * @throws std::domain_error if the mass matrix is invalid
       * @return mass_matrix vector of diagonal values
       */
      Eigen::VectorXd
      validate_diag_mass_matrix(stan::io::var_context& init_mass_matrix,
                                const::size_t num_params,
                                stan::callbacks::writer& error_writer) {
        try {
          init_mass_matrix.validate_dims("inv mass matrix", "mass_matrix",
                                         "vector_d",
                                         init_mass_matrix.to_vec(num_params));
        } catch (const std::domain_error& e) {
          error_writer("Cannot get mass matrix from input file");
          throw std::domain_error("Initialization failure");
        }
        Eigen::VectorXd mass_matrix(num_params);
        std::vector<double> diag_vals(num_params);
        diag_vals = init_mass_matrix.vals_r("mass_matrix");
        for (size_t i = 0; i < num_params; ++i) {
          mass_matrix[i] = diag_vals[i];
        }
        try {
          const char* function = "check_positive_finite";
          const char* name = "inverse_mass_matrix";
          stan::math::check_positive_finite(function, name, mass_matrix);
        } catch (const std::domain_error& e) {
          error_writer("Inverse mass matrix diag vector contains bad value.");
          error_writer("All diagonal elements must be positive and finite.");
          throw std::domain_error("Initialization failure");
        }
        return mass_matrix;
      }

    }
  }
}

#endif
