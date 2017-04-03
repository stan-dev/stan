#ifndef STAN_SERVICES_UTIL_READ_DENSE_MASS_MATRIX_HPP
#define STAN_SERVICES_UTIL_READ_DENSE_MASS_MATRIX_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/math/prim/mat.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace stan {
  namespace services {
    namespace util {

      /**
       * Extract dense mass matrix from a var_context object.
       *
       * @param[in] init_mass_matrix a var_context with array of initial values
       * @param[in] num_params expected number of row, column elements
       * @param[in,out] error_writer message writer
       * @throws std::domain_error if cannot read the mass matrix
       * @return mass_matrix
       */
      Eigen::MatrixXd
      read_dense_mass_matrix(stan::io::var_context& init_mass_matrix,
                                 size_t num_params,
                                 stan::callbacks::writer& error_writer) {
        Eigen::MatrixXd inv_mass_matrix;
        try {
          init_mass_matrix.validate_dims("read dense mass matrix",
                           "mass_matrix", "matrix",
                           init_mass_matrix.to_vec(num_params, num_params));
          std::vector<double> dense_vals =
            init_mass_matrix.vals_r("mass_matrix");
          inv_mass_matrix =
            stan::math::to_matrix(dense_vals, num_params, num_params);
        } catch (const std::exception& e) {
          error_writer("Cannot get mass matrix from input file");
          throw std::domain_error("Initialization failure");
        }

        return inv_mass_matrix;
      }

    }
  }
}

#endif
