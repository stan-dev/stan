#ifndef STAN_MATH_PRIM_SCAL_META_VECTORBUILDER_HPP
#define STAN_MATH_PRIM_SCAL_META_VECTORBUILDER_HPP

#include <stan/math/prim/scal/meta/contains_vector.hpp>
#include <stdexcept>
#include <vector>

namespace stan {

  /**
   *
   *  VectorBuilder allocates type T1 values to be used as
   *  intermediate values. There are 2 template parameters:
   *  - used: boolean variable indicating whether this instance
   *      is used. If this is false, there is no storage allocated
   *      and operator[] throws.
   *  - is_vec: boolean variable indicating whether this instance
   *      should allocate a vector, if it is used. If this is false,
   *      the instance will only allocate a single double value.
   *      If this is true, it will allocate the number requested.
   *      Note that this is calculated based on template parameters
   *      T2 through T7.
   *
   *  These values are mutable.
   */

  template<typename T1, bool used, bool is_vec>
  class VectorBuilderHelper {
  public:
    explicit VectorBuilderHelper(size_t /* n */) { }
    T1& operator[](size_t /* i */) {
      throw std::logic_error("used is false. this should never be called");
    }
  };

  template<typename T1>
  class VectorBuilderHelper<T1, true, false> {
  private:
    T1 x_;
  public:
    explicit VectorBuilderHelper(size_t /* n */) : x_(0.0) { }
    T1& operator[](size_t /* i */) {
      return x_;
    }
  };

  template<typename T1>
  class VectorBuilderHelper<T1, true, true> {
  private:
    std::vector<T1> x_;
  public:
    explicit VectorBuilderHelper(size_t n) : x_(n) { }
    T1& operator[](size_t i) {
      return x_[i];
    }
  };

  template<bool used, typename T1, typename T2, typename T3 = double,
           typename T4 = double, typename T5 = double, typename T6 = double,
           typename T7 = double>
  class VectorBuilder {
  public:
    VectorBuilderHelper<T1, used,
                        contains_vector<T2, T3, T4, T5, T6, T7>::value> a;
    explicit VectorBuilder(size_t n) : a(n) { }
    T1& operator[](size_t i) {
      return a[i];
    }
  };

}
#endif

