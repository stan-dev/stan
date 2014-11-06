#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_POS_SEMIDEFINITE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_POS_SEMIDEFINITE_HPP

#include <sstream>

#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is positive definite
     *
     * NOTE: symmetry is NOT checked by this function
     * 
     * @param function
     * @param y Matrix to test.
     * @param name
     * @return <code>true</code> if the matrix is positive semi-definite.
     * @return throws if any element in y is nan
     * @tparam T Type of scalar.
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
        dom_err(function, name, y(0,0),
                msg.str());
      }
      Eigen::LDLT< Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> > cholesky 
        = y.ldlt();
      if(cholesky.info() != Eigen::Success || (cholesky.vectorD().array() < 0.0).any()) {
        std::ostringstream msg;
        msg << "is not positive semi-definite. " 
            << name << "(0,0) is ";
        dom_err(function, name, y(0,0),
                msg.str());
      }
      for (int i = 0; i < y.size(); i++)
        if (boost::math::isnan(y(i))) {
          std::ostringstream msg;
          msg << "is not positive semi-definite. " 
                  << name << "(0,0) is ";
          dom_err(function, name, y(0,0), 
                  msg.str(), "");
        }
      return true;
    }

  }
}
#endif
