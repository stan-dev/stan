#ifndef STAN__MATH__REV__MAT__FUN__ROWS_DOT_PRODUCT_HPP
#define STAN__MATH__REV__MAT__FUN__ROWS_DOT_PRODUCT_HPP

#include <vector>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/err/check_vector.hpp>
#include <stan/math/prim/mat/err/check_matching_sizes.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/mat/fun/dot_product.hpp>

namespace stan {
  namespace agrad {

    template<typename T1, int R1, int C1, typename T2, int R2, int C2>
    inline
    typename boost::enable_if_c<boost::is_same<T1,var>::value ||
                                boost::is_same<T2,var>::value,
                                Eigen::Matrix<var, R1, 1> >::type
    rows_dot_product(const Eigen::Matrix<T1, R1, C1>& v1,
                     const Eigen::Matrix<T2, R2, C2>& v2) {
      stan::math::check_matching_sizes("dot_product",
                                                 "v1", v1,
                                                 "v2", v2);
      Eigen::Matrix<var, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        ret(j) = var(new dot_product_vari<T1,T2>(v1.row(j),v2.row(j)));
      }
      return ret;
    }
  }
}
#endif
