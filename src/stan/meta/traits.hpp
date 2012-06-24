#ifndef __STAN__META__TRAITS_HPP__
#define __STAN__META__TRAITS_HPP__

#include <boost/type_traits.hpp>

namespace stan {

  namespace agrad {
    class var;
    class vari;
  }

  /**
   * Metaprogramming struct to detect whether a given type is constant
   * in the mathematical sense (not the C++ <code>const</code>
   * sense). If the parameter type is constant, <code>value</code>
   * will be equal to <code>true</code>.
   *
   * The baseline implementation in this abstract base class is to
   * classify a type <code>T</code> as constant if it can be converted
   * (i.e., assigned) to a <code>double</code>.  This baseline should
   * be overridden for any type that should be treated as a variable.
   *
   * @tparam T Type being tested.
   */
  template <typename T>
  struct is_constant {
    /**
     * A boolean constant with equal to <code>true</code> if the
     * type parameter <code>T</code> is a mathematical constant.
     */
    enum { value = boost::is_convertible<T,double>::value };
  };

  template <typename T>
  struct var_to_vi {
    typedef T type;
  };
  template <>
  struct var_to_vi<stan::agrad::var> {
    typedef stan::agrad::vari* type;
  };

  template <typename T>
  struct is_vector {
    enum { value = 0 };
    typedef T type;
  };
  template <typename T>
  struct is_vector <std::vector<T> > {
    enum { value = 1 };
    typedef T type;
  };
  template <typename T>
  struct is_vector <const std::vector<T> > {
    enum { value = 1 };
    typedef T type;
  };

  template <typename T>
  size_t length(const T& x) { 
    if (is_vector<T>::value)
      return ((std::vector<typename is_vector<T>::type>*)&x)->size();
    else
      return 1;
  }

  template <typename T1, typename T2, typename T3>
  size_t max_size(const T1& x1, const T2& x2, const T3& x3) {
    size_t result = length(x1);
    result = result > length(x2) ? result : length(x2);
    result = result > length(x3) ? result : length(x3);
    assert((length(x1) == 1) || (length(x1) == result));
    assert((length(x2) == 1) || (length(x2) == result));
    assert((length(x3) == 1) || (length(x3) == result));
    return result;
  }

  template <typename T, bool is_vec=0>
  class AmbiguousVector {
  private:
    T x_;
  public:
    AmbiguousVector(size_t /*n*/) : x_(0) { }
    T& operator[](int /*i*/) { return x_; }
    size_t size() { return 1; }
  };
  template <typename T>
  class AmbiguousVector<T, 1> {
  private:
    std::vector<T> x_;
  public:
    AmbiguousVector(size_t n) : x_(n, 0) { }
    T& operator[](int i) { return x_[i]; }
    size_t size() { return x_.size(); }
  };

  template<typename T, bool is_vec>
  class VectorView {
  private:
    T x_;
  public:
    VectorView(T x) : x_(x) { }
    T& operator[](int /*i*/) { return x_; }
  };
  template<typename T, bool is_vec>
  class VectorView<std::vector<T>, is_vec> {
  private:
    std::vector<T>* x_;
  public:
    VectorView(std::vector<T>& x) : x_(&x) { }
    T& operator[](int i) { 
      if (is_vec)
        return (*x_)[i];
      else
        return (*x_)[0];
    }
  };
  template<typename T, bool is_vec>
  class VectorView<const std::vector<T>, is_vec> {
  private:
    const std::vector<T>* x_;
  public:
    VectorView(const std::vector<T>& x) : x_(&x) { }
    const T& operator[](int i) const { 
      if (is_vec)
        return (*x_)[i];
      else
        return (*x_)[0];
    }
  };
  template<typename T, bool is_vec>
  class VectorView<T*, is_vec> {
  private:
    T* x_;
  public:
    VectorView(T* x) : x_(x) { }
    T& operator[](int i) { 
      if (is_vec)
        return x_[i];
      else
        return *x_;
    }
  };

}

#endif
