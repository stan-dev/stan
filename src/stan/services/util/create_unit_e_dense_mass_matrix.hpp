#ifndef STAN_SERVICES_UTIL_CREATE_UNIT_E_DENSE_MASS_MATRIX_HPP
#define STAN_SERVICES_UTIL_CREATE_UNIT_E_DENSE_MASS_MATRIX_HPP

#include <stan/io/dump.hpp>
#include <Eigen/Dense>
#include <sstream>

namespace stan {
  namespace services {
    namespace util {

      /**
       * Create a stan::dump object which contains vector "mass_matrix"
       * of specified size where all elements are ones.
       *
       * @param[in] num_params expected number of denseonal elements
       * @return var_context 
       */
      stan::io::dump
      create_unit_e_dense_mass_matrix(size_t num_params) {
        Eigen::MatrixXd inv_mass_matrix(num_params, num_params);
        inv_mass_matrix.setIdentity();
        size_t num_elements = num_params * num_params;
        std::stringstream txt;
        txt << "mass_matrix <- structure(c(";
        for (size_t i = 0; i < num_elements; i++) {
          txt << inv_mass_matrix(i);
          if (i < num_elements - 1)
            txt << ", ";
        }
        txt << "),.Dim=c("
            << num_params
            << ", "
            << num_params
            << "))";
        return stan::io::dump(txt);
      }
    }
  }
}

#endif
