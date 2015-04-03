#ifndef STAN_MATH_PRIM_SCAL_META_VECTORVIEW_HPP
#define STAN_MATH_PRIM_SCAL_META_VECTORVIEW_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stdexcept>
#include <vector>

namespace stan {

  /**
   *  VectorView is a template metaprogram that takes its argument and
   *  allows it to be used like a vector. There are three template parameters
   *  - T: Type of the thing to be wrapped. For example, double, var, vector<double>, etc.
   *  - is_array: Boolean variable indicating whether the underlying type is an array.
   *  - throw_if_accessed: Boolean variable indicating whether this instance should
   *       not be used and should throw if operator[] is used.
   *
   *  For a scalar value, it broadcasts the single value when using
   *  operator[].
   *
   *  For a vector, operator[] looks into the value passed in.
   *  Note: this is not safe. It is possible to read past the size of
   *  an array.
   *
   *  Uses:
   *    Read arguments to prob functions as vectors, even if scalars, so
   *    they can be read by common code (and scalars automatically
   *    broadcast up to behave like vectors) : VectorView of immutable
   *    const array of double* (no allocation)
   *
   *    Build up derivatives into common storage : VectorView of
   *    mutable shared array (no allocation because allocated on
   *    auto-diff arena memory)
   */
  template <typename T,
            bool is_array = stan::is_vector_like<T>::value,
            bool throw_if_accessed = false>
  class VectorView {
  public:
    typedef typename scalar_type<T>::type scalar_t;

    explicit VectorView(scalar_t& c) : x_(&c) { }

    explicit VectorView(std::vector<scalar_t>& v) : x_(&v[0]) { }

    template <int R, int C>
    explicit VectorView(Eigen::Matrix<scalar_t, R, C>& m) : x_(&m(0)) { }

    explicit VectorView(scalar_t* x) : x_(x) { }

    scalar_t& operator[](int i) {
      if (throw_if_accessed)
        throw std::out_of_range("VectorView: this cannot be accessed");
      if (is_array)
        return x_[i];
      else
        return x_[0];
    }
  private:
    scalar_t* x_;
  };


  /**
   *
   *  VectorView that has const correctness.
   */
  template <typename T, bool is_array, bool throw_if_accessed>
  class VectorView<const T, is_array, throw_if_accessed> {
  public:
    typedef typename scalar_type<T>::type scalar_t;

    explicit VectorView(const scalar_t& c) : x_(&c) { }

    explicit VectorView(const scalar_t* x) : x_(x) { }

    explicit VectorView(const std::vector<scalar_t>& v) : x_(&v[0]) { }

    template <int R, int C>
    explicit VectorView(const Eigen::Matrix<scalar_t, R, C>& m) : x_(&m(0)) { }

    const scalar_t& operator[](int i) const {
      if (throw_if_accessed)
        throw std::out_of_range("VectorView: this cannot be accessed");
      if (is_array)
        return x_[i];
      else
        return x_[0];
    }
  private:
    const scalar_t* x_;
  };

  // simplify to hold value in common case where it's more efficient
  template <>
  class VectorView<const double, false, false> {
  public:
    explicit VectorView(double x) : x_(x) { }
    double operator[](int /* i */)  const {
      return x_;
    }
  private:
    const double x_;
  };



}
#endif

