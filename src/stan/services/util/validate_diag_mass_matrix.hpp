#ifndef STAN_SERVICES_UTIL_VALIDATE_DIAG_MASS_MATRIX_HPP
#define STAN_SERVICES_UTIL_VALIDATE_DIAG_MASS_MATRIX_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/math/prim/mat.hpp>

namespace stan {
  namespace services {
    namespace util {

      /**
       * Validate that diag mass matrix is positive definite
       *
       * @param[in] inv_mass_matrix  inverse mass matrix
       * @param[in,out] error_writer message writer
       * @throws std::domain_error if matrix is not positive definite
       */
      void
      validate_diag_mass_matrix(const Eigen::VectorXd& inv_mass_matrix,
                                 stan::callbacks::writer& error_writer) {
        try {
          stan::math::check_finite("check_finite",
                                   "inv_mass_matrix", inv_mass_matrix);
          stan::math::check_positive("check_positive",
                                     "inv_mass_matrix", inv_mass_matrix);
        } catch (const std::domain_error& e) {
          error_writer("Inverse mass matrix not positive definite.");
          throw std::domain_error("Initialization failure");
        }
      }

    }
  }
}

#endif
