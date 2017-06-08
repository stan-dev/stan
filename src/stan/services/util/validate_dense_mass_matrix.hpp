#ifndef STAN_SERVICES_UTIL_VALIDATE_DENSE_MASS_MATRIX_HPP
#define STAN_SERVICES_UTIL_VALIDATE_DENSE_MASS_MATRIX_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim/mat.hpp>

namespace stan {
  namespace services {
    namespace util {

      /**
       * Validate that dense mass matrix is positive definite
       *
       * @param[in] inv_mass_matrix  inverse mass matrix
       * @param[in,out] logger Logger for messages
       * @throws std::domain_error if matrix is not positive definite
       */
      void
      validate_dense_mass_matrix(const Eigen::MatrixXd& inv_mass_matrix,
                                 callbacks::logger& logger) {
        try {
          stan::math::check_pos_definite("check_pos_definite",
                                         "inv_mass_matrix", inv_mass_matrix);
        } catch (const std::domain_error& e) {
          logger.error("Inverse mass matrix not positive definite.");
          throw std::domain_error("Initialization failure");
        }
      }

    }
  }
}

#endif
