#ifndef STAN_MATH_PRIM_MAT_FUN_COMMON_TYPE_HPP
#define STAN_MATH_PRIM_MAT_FUN_COMMON_TYPE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <boost/math/tools/promotion.hpp>
#include <vector>

namespace stan {

  namespace math {

    template <typename T1, typename T2>
    struct common_type {
      typedef typename boost::math::tools::promote_args<T1, T2>::type type;
    };

    template <typename T1, typename T2>
    struct common_type<std::vector<T1>, std::vector<T2> > {
      typedef std::vector<typename common_type<T1, T2>::type> type;
    };

    template <typename T1, typename T2, int R, int C>
    struct common_type<Eigen::Matrix<T1, R, C>, Eigen::Matrix<T2, R, C> > {
      typedef Eigen::Matrix<typename common_type<T1, T2>::type, R, C> type;
    };

  }
}


#endif
