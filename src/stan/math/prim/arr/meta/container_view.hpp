#ifndef STAN_MATH_PRIM_ARR_META_CONTAINER_VIEW_HPP
#define STAN_MATH_PRIM_ARR_META_CONTAINER_VIEW_HPP

#include <stan/math/prim/scal/meta/container_view.hpp>
#include <vector>

namespace stan {

  namespace math {

    template <typename T1, typename T2>
    class container_view<std::vector<T1>, T2> {
      public:
        container_view(const std::vector<T1>& x, T2* y) 
         : y_(y) { }

        T2& operator[](int i) {
          return y_[i];
        }
      private:
        T2* y_;
    };
  }
}

#endif
