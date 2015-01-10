#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_POS_SEMIDEFINITE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_POS_SEMIDEFINITE_HPP

#include <sstream>

#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>

namespace stan {

  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is positive definite
     *
     * NOTE: symmetry is NOT checked by this function
     *
     * @tparam T_y scalar type of the matrix
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     *
     * @return <code>true</code> if the matrix is positive semi-definite.
     * @throw <code>std::domain_error</code> if the matrix is not positive
     *   semi-definite or if any element of the matrix is <code>NaN</code>
     */
    // FIXME: update warnings (message has (0,0) item)
    template <typename T_y>
    inline bool 
    check_pos_semidefinite(const std::string& function,
                           const std::string& name,
                           const Eigen::Matrix<T_y,
                           Eigen::Dynamic,
                           Eigen::Dynamic>& y) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T_y,Dynamic,Dynamic> >::type size_type;

      if (y.rows() == 1 && !(y(0,0) >= 0.0)) {
        std::ostringstream msg;
        msg << "is not positive semi-definite. " 
            << name << "(0,0) is ";
        domain_error(function, name, y(0,0),
                     msg.str());
      }
      Eigen::LDLT< Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> > cholesky 
        = y.ldlt();
      if(cholesky.info() != Eigen::Success || (cholesky.vectorD().array() < 0.0).any()) {
        std::ostringstream msg;
        msg << "is not positive semi-definite. " 
            << name << "(0,0) is ";
        domain_error(function, name, y(0,0),
                     msg.str());
      }
      for (int i = 0; i < y.size(); i++)
        if (boost::math::isnan(y(i))) {
          std::ostringstream msg;
          msg << "is not positive semi-definite. " 
              << name << "(0,0) is ";
          domain_error(function, name, y(0,0), 
                       msg.str(), "");
        }
      return true;
    }

  }
}
#endif
