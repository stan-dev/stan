#ifndef STAN_MATH_PRIM_SCAL_META_CONTAINER_VIEW_HPP
#define STAN_MATH_PRIM_SCAL_META_CONTAINER_VIEW_HPP

#include <stan/math/prim/scal/meta/scalar_type.hpp>

namespace stan {

  namespace math {

    struct dummy { };

    /**
     * Primary template class for container view of
     * array y with same structure as T1 and 
     * size as x
     *
     * operator[](int i) returns reference to view, 
     * indexed by i
     * Specializations handle appropriate broadcasting
     * if size of x is 1
     *
     * Intended for use in OperandsAndPartials
     *
     * @tparam T1 type of view.
     * @tparam T2 type of scalar returned by view.
     * @param x object from which size is to be inferred
     * @param y underlying array 
     */

    template <typename T1, typename T2>
    class container_view {
      public:
        container_view(const T1& x, T2* y) : y_(y) { }

        T2& operator[](int i) {
          return y_[0];
        }
      private:
        T2* y_;
    };

    /**
     * Intended for usage in OperandsAndPartials
     *
     * operator[](int i) 
     * throws exception
     *
     * @tparam T1 type of x
     * @tparam T2 type of scalar returned by view
     * @param x object from which size is to be inferred
     * @param y underlying array 
     */

    template <typename T2>
    class container_view<dummy, T2> {
      public:
        typedef typename stan::scalar_type<T2>::type scalar_t;
        template <typename T1>
        container_view(const T1& x, scalar_t* y){ };
        scalar_t operator[](int n) const {
          throw std::out_of_range("can't access dummy elements.");
        }
    };
  }
}

#endif
