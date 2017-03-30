#ifndef STAN_SERVICES_UTIL_VALIDATE_DENSE_MASS_MATRIX_HPP
#define STAN_SERVICES_UTIL_VALIDATE_DENSE_MASS_MATRIX_HPP

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
       * @throws std::domain_error if the mass matrix is invalid
       * @return mass_matrix vector of diagonal values
       */
      Eigen::MatrixXd
      validate_dense_mass_matrix(stan::io::var_context& init_mass_matrix,
                                const::size_t& num_params,
                                stan::callbacks::writer& error_writer) {
        try {
          init_mass_matrix.validate_dims("validate dense mass matrix", "mass_matrix",
                                         "matrix",
                                         init_mass_matrix.to_vec(num_params,num_params));
        } catch (const std::domain_error& e) {
          error_writer("Cannot get mass matrix from input file");
          throw std::domain_error("Initialization failure");
        }
        size_t num_elements = num_params * num_params;
        std::vector<double> dense_vals(num_elements);
        dense_vals = init_mass_matrix.vals_r("mass_matrix");
        Eigen::MatrixXd inv_mass_matrix(num_params, num_params);
        inv_mass_matrix = stan::math::to_matrix(dense_vals, num_params, num_params);

        try {
          stan::math::check_pos_definite("check_pos_definite", "inv_mass_matrix",
                                         inv_mass_matrix);
        } catch (const std::domain_error& e) {
          error_writer("Inverse mass matrix not positive definite.");
          throw std::domain_error("Initialization failure");
        }
        return inv_mass_matrix;
      }

    }
  }
}

#endif
