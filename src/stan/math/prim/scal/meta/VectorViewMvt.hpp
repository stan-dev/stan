#ifndef STAN__MATH__PRIM__SCAL__META__VECTORVIEWMVT_HPP
#define STAN__MATH__PRIM__SCAL__META__VECTORVIEWMVT_HPP

#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/meta/scalar_type_pre.hpp>
#include <stdexcept>
#include <vector>

namespace stan {

  template <typename T,
            bool is_array 
              = stan::is_vector_like<typename stan::math::value_type<T>::type>::value,
            bool throw_if_accessed = false>
  class VectorViewMvt {
  public: 
    typedef typename scalar_type_pre<T>::type matrix_t;

    VectorViewMvt(matrix_t& m) : x_(&m) { }

    VectorViewMvt(std::vector<matrix_t>& vm) : x_(&vm[0]) { }

    matrix_t& operator[](int i) {
      if (throw_if_accessed) 
        throw std::out_of_range("VectorViewMvt: this cannot be accessed");
      if (is_array) 
        return x_[i];
      else 
        return x_[0];
    }
  private:
    matrix_t* x_;
  };

  /**
   *
   *  VectorViewMvt that has const correctness.
   */
  template <typename T, bool is_array, bool throw_if_accessed>
  class VectorViewMvt<const T, is_array, throw_if_accessed> {
  public: 
    typedef typename scalar_type_pre<T>::type matrix_t;

    VectorViewMvt(const matrix_t& m) : x_(&m) { }

    VectorViewMvt(const std::vector<matrix_t>& vm) : x_(&vm[0]) { }

    const matrix_t& operator[](int i) const {
      if (throw_if_accessed) 
        throw std::out_of_range("VectorViewMvt: this cannot be accessed");
      if (is_array) 
        return x_[i];
      else 
        return x_[0];
    }
  private:
    const matrix_t* x_;
  };


}
#endif

