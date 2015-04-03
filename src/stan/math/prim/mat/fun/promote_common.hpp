#ifndef STAN__MATH__PRIM__MAT__FUN__PROMOTE_COMMON_HPP
#define STAN__MATH__PRIM__MAT__FUN__PROMOTE_COMMON_HPP

#include <stan/math/prim/mat/fun/common_type.hpp>
#include <stan/math/prim/mat/fun/promoter.hpp>

namespace stan {

  namespace math {

    template <typename T1, typename T2, typename F>
    inline
    typename common_type<T1, T2>::type
    promote_common(const F& u) {
      return promoter<F, typename common_type<T1, T2>::type>
        ::promote_to(u);
    }

  }
}


#endif
