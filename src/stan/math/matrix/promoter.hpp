#ifndef STAN__MATH__MATRIX__PROMOTER_HPP
#define STAN__MATH__MATRIX__PROMOTER_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  
  namespace math {
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

  }
}


#endif
