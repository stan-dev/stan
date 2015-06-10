#ifndef STAN_MATH_PRIM_MAT_META_CONTAINER_VIEW_HPP
#define STAN_MATH_PRIM_MAT_META_CONTAINER_VIEW_HPP

#include <stan/math/prim/scal/meta/container_view.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <vector>

namespace stan {

  namespace math {

    /**
     * Template specialization for column vector view of
     * array y with scalar type T2 with size inferred from
     * input column vector x     
     *
     * operator[](int i) returns reference to view 
     * broadcasts as if x is vector<Matrix>
     *
     * Intended for use in OperandsAndPartials
     *
     * @tparam T1 scalar type of input matrix
     * @tparam T2 scalar type returned by view.
     * @param x input matrix
     * @param y underlying array 
     */

    template <typename T1, typename T2>
    class container_view<Eigen::Matrix<T1, -1, 1>, Eigen::Matrix<T2, -1, 1> > {
      public:
        container_view(const Eigen::Matrix<T1, -1, 1>& x, T2* y) 
         : y_(y, x.rows(), x.cols()) { }

        Eigen::Map<Eigen::Matrix<T2, -1, 1> >& operator[](int i) {
          return y_;
        }
      private:
        Eigen::Map<Eigen::Matrix<T2, -1, 1> > y_;
    };

    /**
     * Template specialization for scalar view of
     * array y with scalar type T2 
     * input column vector x     
     *
     * operator[](int i) returns reference to scalar 
     * of type T2 at appropriate index i in array y
     *
     * No bounds checking!
     *
     * Intended for use in OperandsAndPartials
     *
     * @tparam T1 scalar type of input matrix
     * @tparam T2 scalar type returned by view.
     * @param x input matrix
     * @param y underlying array 
     */

    template <typename T1, typename T2>
    class container_view<Eigen::Matrix<T1, -1, 1>, T2> {
      public:
        container_view(const Eigen::Matrix<T1, -1, 1>& x, T2* y) 
         : y_(y) { }

        T2& operator[](int i) {
          return y_[i];
        }
      private:
        T2* y_;
    };

    /**
     * Template specialization for row vector view of
     * array y with scalar type T2 with size inferred from
     * input row vector x     
     *
     * operator[](int i) returns reference to view 
     * broadcasts as if x is vector<Matrix>
     *
     * Intended for use in OperandsAndPartials
     *
     * @tparam T1 scalar type of input matrix
     * @tparam T2 scalar type returned by view.
     * @param x input matrix
     * @param y underlying array 
     */

    template <typename T1, typename T2>
    class container_view<Eigen::Matrix<T1, 1, -1>, Eigen::Matrix<T2, 1, -1> > {
      public:
        container_view(const Eigen::Matrix<T1, 1, -1>& x, T2* y) 
         : y_(y, x.rows(), x.cols()) { }

        Eigen::Map<Eigen::Matrix<T2, 1, -1> >& operator[](int i) {
          return y_;
        }
      private:
        Eigen::Map<Eigen::Matrix<T2, 1, -1> > y_;
    };

    /**
     * Template specialization for scalar view of
     * array y with scalar type T2 
     * input row vector x     
     *
     * operator[](int i) returns reference to scalar 
     * of type T2 at appropriate index i in array y
     *
     * No bounds checking!
     *
     * Intended for use in OperandsAndPartials
     *
     * @tparam T1 scalar type of input matrix
     * @tparam T2 scalar type returned by view.
     * @param x input matrix
     * @param y underlying array 
     */

    template <typename T1, typename T2>
    class container_view<Eigen::Matrix<T1, 1, -1>, T2> {
      public:
        container_view(const Eigen::Matrix<T1, 1, -1>& x, T2* y) 
         : y_(y) { }

        T2& operator[](int i) {
          return y_[i];
        }
      private:
        T2* y_;
    };

    /**
     * Template specialization for matrix view of
     * array y with scalar type T2 with rows and columns
     * inferred from input matrix x     
     *
     * operator[](int i) returns reference to view 
     * broadcasts as if x is vector<Matrix>
     *
     * Intended for use in OperandsAndPartials
     *
     * @tparam T1 scalar type of input matrix
     * @tparam T2 scalar type returned by view.
     * @param x input matrix
     * @param y underlying array 
     */

    template <typename T1, typename T2, int M, int N>
    class container_view<Eigen::Matrix<T1, M, N>, Eigen::Matrix<T2, M, N> > {
      public:
        container_view(const Eigen::Matrix<T1, M, N>& x, T2* y) 
         : y_(y, x.rows(), x.cols()) { }

        Eigen::Map<Eigen::Matrix<T2, M, N> >& operator[](int i) {
          return y_;
        }
      private:
        Eigen::Map<Eigen::Matrix<T2, M, N> > y_;
    };

    /**
     * Template specialization for matrix view of
     * array y with scalar type T2 with proper indexing
     * inferred from input vector of matrices x     
     *
     * operator[](int i) returns reference to view 
     * as if indexed at i 
     *
     * No bounds checking!
     *
     * Intended for use in OperandsAndPartials
     *
     * @tparam T1 scalar type of input matrix
     * @tparam T2 scalar type returned by view.
     * @param x input matrix
     * @param y underlying array 
     */

    template <typename T1, typename T2, int M, int N>
    class container_view<std::vector<Eigen::Matrix<T1, M, N> >,
                          Eigen::Matrix<T2, M, N> > {
      public:
        container_view(const std::vector<Eigen::Matrix<T1, M, N> >& x, T2* y) 
         : y_view(y, 1, 1), y_(y) {
           if(x.size() > 0) {
             rows = x[0].rows();
             cols = x[0].cols();
           }
         }

        Eigen::Map<Eigen::Matrix<T2, M, N> >& operator[](int i) {
          int offset = i * rows * cols; 
          new (&y_view) Eigen::Map<Eigen::Matrix<T2, M, N> >(y_ + offset, rows, cols); 
          return y_view;
        }
      private:
        Eigen::Map<Eigen::Matrix<T2, M, N> > y_view;
        T2* y_;
        int rows;
        int cols;
    };
  }
}
#endif
