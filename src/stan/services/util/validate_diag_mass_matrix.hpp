#ifndef STAN_SERVICES_UTIL_VALIDATE_DIAG_MASS_MATRIX_HPP
#define STAN_SERVICES_UTIL_VALIDATE_DIAG_MASS_MATRIX_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim/mat.hpp>

namespace stan {
  namespace services {
    namespace util {

      /**
       * Validate that diag mass matrix is positive definite
       *
       * @param[in] inv_mass_matrix  inverse mass matrix
       * @param[in,out] logger Logger for messages
       * @throws std::domain_error if matrix is not positive definite
       */
      void
      validate_diag_mass_matrix(const Eigen::VectorXd& inv_mass_matrix,
                                callbacks::logger& logger) {
        try {
          stan::math::check_finite("check_finite",
                                   "inv_mass_matrix", inv_mass_matrix);
          stan::math::check_positive("check_positive",
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
