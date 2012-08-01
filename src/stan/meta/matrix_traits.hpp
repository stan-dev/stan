#ifndef __STAN__META__MATRIX_TRAITS_HPP__
#define __STAN__META__MATRIX_TRAITS_HPP__

#include <stan/math/matrix.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

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
  size_t length(const Eigen::Matrix<T,Eigen::Dynamic,1>& v) {
    return v.size();
  }
  template <typename T>
  size_t length(const Eigen::Matrix<T,1,Eigen::Dynamic>& rv) {
    return rv.size();
  }

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

}

#endif
