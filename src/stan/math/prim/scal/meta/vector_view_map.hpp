#ifndef STAN_MATH_PRIM_SCAL_META_VECTOR_VIEW_MAP_HPP
#define STAN_MATH_PRIM_SCAL_META_VECTOR_VIEW_MAP_HPP

#include <stan/math/prim/scal/meta/scalar_type.hpp>

namespace stan {

  namespace math {

    struct dummy { };

    template <typename T1, typename T2>
    class vector_view_map {
      public:
        vector_view_map(const T1& x, T2* y) : y_(y) { }

        T2& operator[](int i) {
          return y_[0];
        }
      private:
        T2* y_;
    };

    template <typename T2>
    class vector_view_map<dummy, T2> {
      public:
        typedef typename stan::scalar_type<T2>::type scalar_t;
        template <typename T1>
        vector_view_map(const T1& x, scalar_t* y){ };
        scalar_t operator[](int n) const {
          throw std::out_of_range("can't access dummy elements.");
        }
    };
  }
}

#endif
