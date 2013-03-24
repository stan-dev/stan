#ifndef __STAN__MATH__MATRIX_HPP__
#define __STAN__MATH__MATRIX_HPP__

#include <stdarg.h>
#include <stdexcept>
#include <ostream>
#include <vector>

#include <boost/math/tools/promotion.hpp>

#define EIGEN_DENSEBASE_PLUGIN "stan/math/EigenDenseBaseAddons.hpp"
#include <Eigen/Dense>
#include <Eigen/QR>

#include <stan/math/boost_error_handling.hpp>

namespace stan {
  
  namespace math {
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type size_type;
    // from input type F to output type T 

    // scalar, F != T  (base template)
    template <typename F, typename T>
    struct promoter {
      inline static void promote(const F& u, T& t) {
        t = u;
      }
      inline static T promote_to(const F& u) {
        return u;
      }
    };
    // scalar, F == T
    template <typename T>
    struct promoter<T,T> {
      inline static void promote(const T& u, T& t) {
        t = u;
      }
      inline static T promote_to(const T& u) {
        return u;
      }
    };

    // std::vector, F != T
    template <typename F, typename T>
    struct promoter<std::vector<F>, std::vector<T> > {
      inline static void promote(const std::vector<F>& u,
                          std::vector<T>& t) {
        t.resize(u.size());
        for (size_t i = 0; i < u.size(); ++i)
          promoter<F,T>::promote(u[i],t[i]);
      }
      inline static std::vector<T>
      promote_to(const std::vector<F>& u) {
        std::vector<T> t;
        promoter<std::vector<F>,std::vector<T> >::promote(u,t);
        return t;
      }
    };
    // std::vector, F == T
    template <typename T>
    struct promoter<std::vector<T>, std::vector<T> > {
      inline static void promote(const std::vector<T>& u,
                          std::vector<T>& t) {
        t = u;
      }
      inline static std::vector<T> promote_to(const std::vector<T>& u) {
        return u;
      }
    };

    // Eigen::Matrix, F != T
    template <typename F, typename T, int R, int C>
    struct promoter<Eigen::Matrix<F,R,C>, Eigen::Matrix<T,R,C> > {
      inline static void promote(const Eigen::Matrix<F,R,C>& u,
                          Eigen::Matrix<T,R,C>& t) {
        t.resize(u.rows(), u.cols());
        for (int i = 0; i < u.size(); ++i)
          promoter<F,T>::promote(u(i),t(i));
      }
      inline static Eigen::Matrix<T,R,C>
      promote_to(const Eigen::Matrix<F,R,C>& u) {
        Eigen::Matrix<T,R,C> t;
        promoter<Eigen::Matrix<F,R,C>,Eigen::Matrix<T,R,C> >::promote(u,t);
        return t;
      }
    };
    // Eigen::Matrix, F == T
    template <typename T, int R, int C>
    struct promoter<Eigen::Matrix<T,R,C>, Eigen::Matrix<T,R,C> > {
      inline static void promote(const Eigen::Matrix<T,R,C>& u,
                          Eigen::Matrix<T,R,C>& t) {
        t = u;
      }
      inline static Eigen::Matrix<T,R,C> promote_to(const Eigen::Matrix<T,R,C>& u) {
        return u;
      }
    };

    template <typename T1, typename T2>
    struct common_type {
      typedef typename boost::math::tools::promote_args<T1,T2>::type type;
    };

    template <typename T1, typename T2>
    struct common_type<std::vector<T1>, std::vector<T2> > {
      typedef std::vector<typename common_type<T1,T2>::type> type;
    };
    
    template <typename T1, typename T2, int R, int C>
    struct common_type<Eigen::Matrix<T1,R,C>, Eigen::Matrix<T2,R,C> > {
      typedef Eigen::Matrix<typename common_type<T1,T2>::type,R,C> type;
    };

    template <typename T1, typename T2, typename F>
    inline
    typename common_type<T1,T2>::type
    promote_common(const F& u) {
      return promoter<F, typename common_type<T1,T2>::type>
        ::promote_to(u);
    }




    /**
     * Structure for building up arrays in an expression (rather than
     * in statements) using an argumentchaining add() method and 
     * a getter method array() to return the result.
     */
    template <typename T>
    struct array_builder {
      std::vector<T> x_;
      array_builder() : x_() { }
      template <typename F>
      array_builder& add(const F& u) {
        T t;
        promoter<F,T>::promote(u,t);
        x_.push_back(t);
        return *this;
      }
      std::vector<T> array() {
        return x_;
      }
    };

    /**
     * Type for matrix of double values.
     */
    typedef 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
    matrix_d;

    /**
     * Type for (column) vector of double values.
     */
    typedef 
    Eigen::Matrix<double,Eigen::Dynamic,1>
    vector_d;

    /**
     * Type for (row) vector of double values.
     */
    typedef 
    Eigen::Matrix<double,1,Eigen::Dynamic>
    row_vector_d;

    namespace {

      template <typename T>
      void resize(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, 
                  const std::vector<size_t>& dims, 
                  size_t pos) {
        x.resize(dims[pos],dims[pos+1]);
      }

      template <typename T>
      void resize(Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                  const std::vector<size_t>& dims, 
                  size_t pos) {
        x.resize(dims[pos]);
      }

      template <typename T>
      void resize(Eigen::Matrix<T,1,Eigen::Dynamic>& x, 
                  const std::vector<size_t>& dims, 
                  size_t pos) {
        x.resize(dims[pos]);
      }


      void resize(double /*x*/, 
                  const std::vector<size_t>& /*dims*/, 
                  size_t /*pos*/) {
        // no-op
      }

      template <typename T>
      void resize(std::vector<T>& x, 
                  const std::vector<size_t>& dims, 
                  size_t pos) {
        x.resize(dims[pos]);
        ++pos;
        if (pos >= dims.size()) return; // skips lowest loop to scalar
        for (size_t i = 0; i < x.size(); ++i)
          resize(x[i],dims,pos);
      }

    }

    /**
     * Recursively resize the specified vector of vectors,
     * which must bottom out at scalar values, Eigen vectors
     * or Eigen matrices.
     *
     * @param x Array-like object to resize.
     * @param dims New dimensions.
     * @tparam T Type of object being resized.
     */
    template <typename T>
    inline void resize(T& x, std::vector<size_t> dims) {
      resize(x,dims,0U);
    }

    // polymorphic gets with bounds checking



    // void eigen_decompose_sym(const matrix_d& m,
    //                          vector_d& eigenvalues,
    //                          matrix_d& eigenvectors) {
    //   Eigen::SelfAdjointEigenSolver<matrix_d> solver(m);
    //   eigenvalues = solver.eigenvalues().real();
    //   eigenvectors = solver.eigenvectors().real();
    // }


    // void svd(const matrix_d& m, matrix_d& u, matrix_d& v, vector_d& s) {
    //   static const unsigned int THIN_SVD_OPTIONS
    //     = Eigen::ComputeThinU | Eigen::ComputeThinV;
    //   Eigen::JacobiSVD<matrix_d> svd(m, THIN_SVD_OPTIONS);
    //   u = svd.matrixU();
    //   v = svd.matrixV();
    //   s = svd.singularValues();
    // }



  }
}
#endif

