#ifndef STAN_MATH_PRIM_ARR_META_VECTOR_VIEW_MAP_HPP
#define STAN_MATH_PRIM_ARR_META_VECTOR_VIEW_MAP_HPP

#include <stan/math/prim/scal/meta/vector_view_map.hpp>
#include <vector>

template <typename T1, typename T2>
class vector_view_map<std::vector<T1>, T2> {
  public:
    vector_view_map(const std::vector<T1>& x, T2* y) 
     : y_(y) { }

    T2& operator[](int i) {
      return y_[i];
    }
  private:
    T2* y_;
};

#endif
