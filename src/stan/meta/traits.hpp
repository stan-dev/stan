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

  // FIXME: use boost::type_traits::remove_all_extents to extend to array/ptr types

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
    /** 
     * Base scalar type for object.
     */
    typedef T type;
  };

  /**
   * Metaprogram specialization extracting the base type of
   * a standard vector recursively.
   *
   * @tparam Scalar type of vector.
   */
  template <typename T>
  struct scalar_type<std::vector<T> > {
    typedef typename scalar_type<T>::type type;
  };

  template <typename T>
  struct scalar_type<Eigen::Matrix<T,Eigen::Dynamic,1> > {
    typedef typename scalar_type<T>::type type;
  };
  template <typename T>
  struct scalar_type<Eigen::Matrix<T,1,Eigen::Dynamic> > {
    typedef typename scalar_type<T>::type type;
  };
  template <typename T>
  struct scalar_type<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > {
    typedef typename scalar_type<T>::type type;
  };

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

  // Matt's original version
  // size_t length(const T& x) { 
  //   if (is_vector<T>::value)
  //     return ((std::vector<typename is_vector<T>::type>*)&x)->size();
  //   else
  //     return 1;
  // }

  // FIXME: not a trait, move to meta

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


  template <typename T>
  size_t size_of(const T& x) {
    return 1U;
  }
  template <typename T>
  size_t size_of(const std::vector<T>& x) {
    return x.size();
  }
  template <typename T>
  size_t size_of(const Eigen::Matrix<T,Eigen::Dynamic,1>& v) {
    return v.size();
  }
  template <typename T>
  size_t size_of(const Eigen::Matrix<T,1,Eigen::Dynamic>& rv) {
    return rv.size();
  }
  template <typename T>
  size_t size_of(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
    return m.size();
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

  // AmbiguousVector is the simple VectorView for writing doubles into
  // should work for Eigen vectors and std::vector
  // FIXME:  used only with T= double, so could just fix it.
  // FIXME:  rename -- "Ambiguous" is not the right word here
  template <typename T, bool is_vec = 0>
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


  // two template params for use in partials_vari OperandsAndPartials
  template<typename T, bool is_vec = stan::is_vector<T>::value>
  class VectorView {
  private:
    T x_;
  public:
    VectorView(T x) : x_(x) { }
    T& operator[](int /*i*/) { 
      return x_; 
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

  template<typename T>
  class VectorView<std::vector<T>, true> {
  private:
    std::vector<T>& x_;
  public:
    VectorView(std::vector<T>& x) : x_(x) { }
    T& operator[](int i) { 
      return x_[i];
    }
  };
  template<typename T>
  class VectorView<const std::vector<T>, true> {
  private:
    const std::vector<T>& x_;
  public:
    VectorView(const std::vector<T>& x) : x_(x) { }
    const T& operator[](int i) const { 
      return x_[i];
    }
  };


  template<typename T>
  class VectorView<Eigen::Matrix<T,Eigen::Dynamic,1>, true> {
  private:
    Eigen::Matrix<T,Eigen::Dynamic,1>& x_;
  public:
    VectorView(Eigen::Matrix<T,Eigen::Dynamic,1>& x) : x_(x) { }
    T& operator[](int i) { 
      return x_[i];
    }
  };
  template<typename T>
  class VectorView<const Eigen::Matrix<T,Eigen::Dynamic,1>, true> {
  private:
    const Eigen::Matrix<T,Eigen::Dynamic,1>& x_;
  public:
    VectorView(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) : x_(x) { }
    const T& operator[](int i) { 
      return x_[i];
    }
  };

  template<typename T>
  class VectorView<Eigen::Matrix<T,1,Eigen::Dynamic>, true> {
  private:
    Eigen::Matrix<T,1,Eigen::Dynamic>& x_;
  public:
    VectorView(Eigen::Matrix<T,1,Eigen::Dynamic>& x) : x_(x) { }
    T& operator[](int i) { 
      return x_[i];
    }
  };
  template<typename T>
  class VectorView<const Eigen::Matrix<T,1,Eigen::Dynamic>, true> {
  private:
    const Eigen::Matrix<T,1,Eigen::Dynamic>& x_;
  public:
    VectorView(const Eigen::Matrix<T,1,Eigen::Dynamic>& x) : x_(x) { }
    const T& operator[](int i) { 
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
