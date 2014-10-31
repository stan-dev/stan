#ifndef STAN__META__TRAITS_HPP
#define STAN__META__TRAITS_HPP


#include <vector>
#include <boost/type_traits.hpp>
#include <boost/type_traits/is_arithmetic.hpp> 
#include <boost/math/tools/promotion.hpp>

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/value_type.hpp>
#include <stan/math/meta/value_type.hpp>

namespace stan {

  struct error_index {
    enum { value = 
#ifdef ERROR_INDEX
ERROR_INDEX
#else
1
#endif
    };
  };

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

  template <typename T>
  struct is_constant_struct<Eigen::Block<T> > {
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
  template <typename T>
  struct is_vector<Eigen::Block<T> > {
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
      typedef typename 
      scalar_type_helper<is_vector<typename stan::math::value_type<T>::type>::value, 
                         typename stan::math::value_type<T>::type>::type 
      type;
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

  template <typename T>
  inline T get(const T& x, size_t n) {
    return x;
  }
  template <typename T>
  inline T get(const std::vector<T>& x, size_t n) {
    return x[n];
  }
  template <typename T, int R, int C>
  inline T get(const Eigen::Matrix<T,R,C>& m, size_t n) {
    return m(static_cast<int>(n));
  }

  

  // length() should only be applied to primitive or std vector or Eigen vector
  template <typename T>
  size_t length(const T& /*x*/) {
    return 1U;
  }
  template <typename T>
  size_t length(const std::vector<T>& x) {
    return x.size();
  }
  template <typename T, int R, int C>
  size_t length(const Eigen::Matrix<T,R,C>& m) {
    return m.size();
  }

  template<typename T, bool is_vec>
  struct size_of_helper {
    static size_t size_of(const T& /*x*/) {
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
    return result;
  }

  template <typename T1, typename T2, typename T3>
  size_t max_size(const T1& x1, const T2& x2, const T3& x3) {
    size_t result = length(x1);
    result = result > length(x2) ? result : length(x2);
    result = result > length(x3) ? result : length(x3);
    return result;
  }

  template <typename T1, typename T2, typename T3, typename T4>
  size_t max_size(const T1& x1, const T2& x2, const T3& x3, const T4& x4) {
    size_t result = length(x1);
    result = result > length(x2) ? result : length(x2);
    result = result > length(x3) ? result : length(x3);
    result = result > length(x4) ? result : length(x4);
    return result;
  }

  // ****************** additions for new VV *************************
  template <typename T>
  struct scalar_type<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > {
    typedef typename scalar_type<T>::type type;
  };

  template <typename T>
  struct scalar_type<T*> {
    typedef typename scalar_type<T>::type type;
  };


  // handles scalar, eigen vec, eigen row vec, std vec
  template <typename T>
  struct is_vector_like {
    enum { value = stan::is_vector<T>::value };  
  };
  template <typename T>
  struct is_vector_like<T*> {
    enum { value = true };
  };
  // handles const
  template <typename T>
  struct is_vector_like<const T> {
    enum { value = stan::is_vector_like<T>::value };  
  };
  // handles eigen matrix
  template <typename T>
  struct is_vector_like<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > {
    enum { value = true };
  };


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

    VectorView(scalar_t& c) : x_(&c) { }

    VectorView(std::vector<scalar_t>& v) : x_(&v[0]) { }

    template <int R, int C>
    VectorView(Eigen::Matrix<scalar_t,R,C>& m) : x_(&m(0)) { }

    VectorView(scalar_t* x) : x_(x) { }

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

    VectorView(const scalar_t& c) : x_(&c) { }

    VectorView(const scalar_t* x) : x_(x) { }

    VectorView(const std::vector<scalar_t>& v) : x_(&v[0]) { }

    template <int R, int C>
    VectorView(const Eigen::Matrix<scalar_t,R,C>& m) : x_(&m(0)) { }

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
    VectorView(double x) : x_(x) { }
    double operator[](int /* i */)  const {
      return x_;
    }
  private:
    const double x_;
  };


  /**
   *
   *  DoubleVectorView allocates double values to be used as
   *  intermediate values. There are 2 template parameters:
   *  - used: boolean variable indicating whether this instance
   *      is used. If this is false, there is no storage allocated
   *      and operator[] throws.
   *  - is_vec: boolean variable indicating whether this instance
   *      should allocate a vector, if it is used. If this is false,
   *      the instance will only allocate a single double value.
   *      If this is true, it will allocate the number requested.
   *
   *  These values are mutable.
   */
  template<bool used, bool is_vec>
  class DoubleVectorView {
  public:
    DoubleVectorView(size_t /* n */) { }
    double& operator[](size_t /* i */) {
      throw std::runtime_error("used is false. this should never be called");
    }
  };

  template<>
  class DoubleVectorView<true, false> {
  private:
    double x_;
  public:
    DoubleVectorView(size_t /* n */) : x_(0.0) { }
    double& operator[](size_t /* i */) {
      return x_;
    }
  };

  template<>
  class DoubleVectorView<true, true> {
  private:
    std::vector<double> x_;
  public:
    DoubleVectorView(size_t n) : x_(n) { }
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


  template <typename T>
  struct is_fvar {
    enum { value = false };
  };
  template <typename T>
  struct is_fvar<stan::agrad::fvar<T> > {
    enum { value = true };
  };


  template <typename T>
  struct is_var {
    enum { value = false };
  };
  template <>
  struct is_var<stan::agrad::var> {
    enum { value = true };
  };



  // FIXME:  pull out scalar types

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
    struct contains_fvar {
      enum {
        value = is_fvar<typename scalar_type<T1>::type>::value
        || is_fvar<typename scalar_type<T2>::type>::value
        || is_fvar<typename scalar_type<T3>::type>::value
        || is_fvar<typename scalar_type<T4>::type>::value
        || is_fvar<typename scalar_type<T5>::type>::value
        || is_fvar<typename scalar_type<T6>::type>::value
      };
    };


    template <typename T1, 
              typename T2 = double, 
              typename T3 = double, 
              typename T4 = double, 
              typename T5 = double, 
              typename T6 = double>
    struct is_var_or_arithmetic {
      enum {
        value 
        = (is_var<typename scalar_type<T1>::type>::value 
           || boost::is_arithmetic<typename scalar_type<T1>::type>::value)
        && (is_var<typename scalar_type<T2>::type>::value 
            || boost::is_arithmetic<typename scalar_type<T2>::type>::value)
        && (is_var<typename scalar_type<T3>::type>::value 
            || boost::is_arithmetic<typename scalar_type<T3>::type>::value)
        && (is_var<typename scalar_type<T4>::type>::value
            || boost::is_arithmetic<typename scalar_type<T4>::type>::value)
        && (is_var<typename scalar_type<T5>::type>::value 
            || boost::is_arithmetic<typename scalar_type<T5>::type>::value)
        && (is_var<typename scalar_type<T6>::type>::value
            || boost::is_arithmetic<typename scalar_type<T6>::type>::value)
      };
    };

  namespace {
    template <bool is_vec, typename T, typename T_container>
    struct scalar_type_helper_pre {
      typedef T_container type;
    };
    
    template <typename T, typename T_container> 
    struct scalar_type_helper_pre<true, T, T_container> {
      typedef typename 
      scalar_type_helper_pre<is_vector<typename stan::math::value_type<T>::type>::value, 
                             typename stan::math::value_type<T>::type, 
                             typename stan::math::value_type<T_container>::type>::type 
      type;
    };
  }
  
  /**
    * Metaprogram structure to determine the type of first container of
    * the base scalar type of a template argument.
    *
    * @tparam T Type of object.
  */
  template <typename T>
  struct scalar_type_pre {
    typedef typename 
    scalar_type_helper_pre<is_vector<typename stan::math::value_type<T>::type>::value,
                           typename stan::math::value_type<T>::type, T>::type 
    type;
  };


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

  // length_mvt() should only be applied to std vector or Eigen matrix
  template <typename T>
  size_t length_mvt(const T& ) {
    throw std::out_of_range("length_mvt passed to an unrecognized type.");
    return 1U;
  }
  template <typename T, int R, int C>
  size_t length_mvt(const Eigen::Matrix<T,R,C>& ) {
    return 1U;
  }
  template <typename T, int R, int C>
  size_t length_mvt(const std::vector<Eigen::Matrix<T,R,C> >& x) {
    return x.size();
  }

  template <typename T1, typename T2>
  size_t max_size_mvt(const T1& x1, const T2& x2) {
    size_t result = length_mvt(x1);
    result = result > length_mvt(x2) ? result : length_mvt(x2);
    return result;
  }

  template <typename T1, typename T2, typename T3>
  size_t max_size_mvt(const T1& x1, const T2& x2, const T3& x3) {
    size_t result = length_mvt(x1);
    result = result > length_mvt(x2) ? result : length_mvt(x2);
    result = result > length_mvt(x3) ? result : length_mvt(x3);
    return result;
  }

  template <typename T1, typename T2, typename T3, typename T4>
  size_t max_size_mvt(const T1& x1, const T2& x2, const T3& x3, const T4& x4) {
    size_t result = length_mvt(x1);
    result = result > length_mvt(x2) ? result : length_mvt(x2);
    result = result > length_mvt(x3) ? result : length_mvt(x3);
    result = result > length_mvt(x4) ? result : length_mvt(x4);
    return result;
  }

}
#endif

