#ifndef __STAN__META__TRAITS_HPP__
#define __STAN__META__TRAITS_HPP__

#include <vector>
#include <boost/type_traits.hpp>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix.hpp>

namespace stan {

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


  /**
   * Metaprogram to determine if a type has a base scalar
   * type that can be assigned to type double.
   */
  template <typename T>
  struct is_constant_struct {
    enum { value = is_constant<T>::value };
  };


  template <typename T>
  struct is_constant_struct<std::vector<T> > {
    enum { value = is_constant_struct<T>::value };
  };

  template <typename T, int R, int C>
  struct is_constant_struct<Eigen::Matrix<T,R,C> > {
    enum { value = is_constant_struct<T>::value };
  };


  // FIXME: use boost::type_traits::remove_all_extents to extend to array/ptr types

  template <typename T>
  struct is_vector {
    enum { value = 0 };
    typedef T type;
  };
  template <typename T>
  struct is_vector<const T> {
    enum { value = is_vector<T>::value };
    typedef T type;
  };
  template <typename T>
  struct is_vector<std::vector<T> > {
    enum { value = 1 };
    typedef T type;
  };
  
  template <typename T>
  struct is_vector<Eigen::Matrix<T,Eigen::Dynamic,1> > {
    enum { value = 1 };
    typedef T type;
  };
  template <typename T>
  struct is_vector<Eigen::Matrix<T,1,Eigen::Dynamic> > {
    enum { value = 1 };
    typedef T type;
  };

  namespace {
    template <bool is_vec, typename T>
    struct scalar_type_helper {
      typedef T type;
    };
    
    template <typename T> 
    struct scalar_type_helper<true, T> {
      typedef typename scalar_type_helper<is_vector<typename T::value_type>::value, typename T::value_type>::type type;
    };
  }
  /**
   * Metaprogram structure to determine the base scalar type
   * of a template argument.
   *
   * <p>This base class should be specialized for structured types.
   *
   * @tparam T Type of object.
   */
  template <typename T>
  struct scalar_type {
    typedef typename scalar_type_helper<is_vector<T>::value, T>::type type;
  };

  // length() should only be applied to primitive or std vector or Eigen vector
  template <typename T>
  size_t length(const T& x) {
    return 1U;
  }
  template <typename T>
  size_t length(const std::vector<T>& x) {
    return x.size();
  }
  
  template <typename T>
  size_t length(const Eigen::Matrix<T,Eigen::Dynamic,1>& v) {
    return v.size();
  }
  template <typename T>
  size_t length(const Eigen::Matrix<T,1,Eigen::Dynamic>& rv) {
    return rv.size();
  }

  template<typename T, bool is_vec>
  struct size_of_helper {
    static size_t size_of(const T& x) {
      return 1U;
    }
  };

  template<typename T>
  struct size_of_helper<T, true> {
    static size_t size_of(const T& x) {
      return x.size();
    }
  };

  template <typename T>
  size_t size_of(const T& x) {
    return size_of_helper<T, is_vector<T>::value>::size_of(x);
  }

  template <typename T1, typename T2>
  size_t max_size(const T1& x1, const T2& x2) {
    size_t result = length(x1);
    result = result > length(x2) ? result : length(x2);
    // assert((length(x1) == 1) || (length(x1) == result));
    // assert((length(x2) == 1) || (length(x2) == result));
    return result;
  }

  template <typename T1, typename T2, typename T3>
  size_t max_size(const T1& x1, const T2& x2, const T3& x3) {
    size_t result = length(x1);
    result = result > length(x2) ? result : length(x2);
    result = result > length(x3) ? result : length(x3);
    // assert((length(x1) == 1) || (length(x1) == result));
    // assert((length(x2) == 1) || (length(x2) == result));
    // assert((length(x3) == 1) || (length(x3) == result));
    return result;
  }
  
  template<typename T, 
	   bool is_vec = stan::is_vector<T>::value>
  class VectorView {
  private:
    T* x_;
  public:
    VectorView(T& x) : x_(&x) { }
    typename scalar_type<T>::type& operator[](int /*i*/) {
      return *x_;
    }
  };
  
  template<typename T>
  class VectorView<T*,false> {
  private:
    T* x_;
  public:
    VectorView(T* x) : x_(x) { }
    typename scalar_type<T>::type& operator[](int i) {
      return *x_;
    }
  };
  
  template<typename T>
  class VectorView<T,true> {
  private:
    T* x_;
  public:
    VectorView(T& x) : x_(&x) { }
    typename scalar_type<T>::type& operator[](int i) {
      return (*x_)[i];
    }
  };
  
  template<typename T>
  class VectorView<T*,true> {
  private:
    T* x_;
  public:
    VectorView(T* x) : x_(x) { }
    typename scalar_type<T>::type& operator[](int i) {
      return x_[i];
    }
  };

  template<typename T>
  class VectorView<const T,true> {
  private:
    const T* x_;
  public:
    VectorView(const T& x) : x_(&x) { }
    typename scalar_type<T>::type operator[](int i) {
      return (*x_)[i];
    }
  };

  

  template<bool used, typename T, bool is_vec = stan::is_vector<T>::value>
  class DoubleVectorView {
  public:
    DoubleVectorView(const T& /* x */) { }
    double& operator[](size_t /* i */) {
      throw std::runtime_error("used is false. this should never be called");
    }
  };

  template<typename T>
  class DoubleVectorView<true, T, false> {
  private:
    double x_;
  public:
    DoubleVectorView(const T& x) : x_(0.0) { }
    double& operator[](size_t /* i */) {
      return x_;
    }
  };

  template<typename T>
  class DoubleVectorView<true, T, true> {
  private:
    std::vector<double> x_;
  public:
    DoubleVectorView(const T& x) : x_(length(x)) { }
    double& operator[](size_t i) {
      return x_[i];
    }
  };

  /**
   * Metaprogram to calculate the base scalar return type resulting
   * from promoting all the scalar types of the template parameters.
   */
    template <typename T1, 
              typename T2 = double, 
              typename T3 = double, 
              typename T4 = double, 
              typename T5 = double, 
              typename T6 = double>
    struct return_type {
      typedef typename 
      boost::math::tools::promote_args<typename scalar_type<T1>::type,
                                       typename scalar_type<T2>::type,
                                       typename scalar_type<T3>::type,
                                       typename scalar_type<T4>::type,
                                       typename scalar_type<T5>::type,
                                       typename scalar_type<T6>::type>::type
      type;
    };



}

#endif
