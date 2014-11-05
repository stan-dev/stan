#ifndef STAN__MATH__MATRIX__INVERSE_SPD_HPP
#define STAN__MATH__MATRIX__INVERSE_SPD_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the inverse of the specified symmetric, pos/neg-definite matrix.
     * @param m Specified matrix.
     * @return Inverse of the matrix.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    inverse_spd(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      stan::error_handling::check_square("inverse_spd", "m", m);
      stan::error_handling::check_symmetric("inverse_spd", "m", m);
      Eigen::LDLT< Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt(0.5*(m+m.transpose()));
      if (ldlt.info() != Eigen::Success)
        throw std::domain_error("Error in inverse_spd, LDLT factorization failed");
      if (!ldlt.isPositive() || (ldlt.vectorD().array() <= 0).any())
        throw std::domain_error("Error in inverse_spd, matrix not positive definite");
      return ldlt.solve(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Identity(m.rows(),m.cols()));
    }

  }
}
#endif
