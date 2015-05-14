#ifndef STAN_MATH_PRIM_MAT_META_VECTOR_VIEW_MAP_HPP
#define STAN_MATH_PRIM_MAT_META_VECTOR_VIEW_MAP_HPP

#include <stan/math/prim/scal/meta/vector_view_map.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <vector>

namespace stan {

  namespace math {

    template <typename T1, typename T2>
    class vector_view_map<Eigen::Matrix<T1, -1, 1>, Eigen::Matrix<T2, -1, 1> > {
      public:
        vector_view_map(const Eigen::Matrix<T1, -1, 1>& x, T2* y) 
         : y_(y, x.rows(), x.cols()) { }

        Eigen::Map<Eigen::Matrix<T2, -1, 1> >& operator[](int i) {
          return y_;
        }
      private:
        Eigen::Map<Eigen::Matrix<T2, -1, 1> > y_;
    };

    template <typename T1, typename T2>
    class vector_view_map<Eigen::Matrix<T1, -1, 1>, T2> {
      public:
        vector_view_map(const Eigen::Matrix<T1, -1, 1>& x, T2* y) 
         : y_(y) { }

        T2& operator[](int i) {
          return y_[i];
        }
      private:
        T2* y_;
    };

    template <typename T1, typename T2>
    class vector_view_map<Eigen::Matrix<T1, 1, -1>, Eigen::Matrix<T2, 1, -1> > {
      public:
        vector_view_map(const Eigen::Matrix<T1, 1, -1>& x, T2* y) 
         : y_(y, x.rows(), x.cols()) { }

        Eigen::Map<Eigen::Matrix<T2, 1, -1> >& operator[](int i) {
          return y_;
        }
      private:
        Eigen::Map<Eigen::Matrix<T2, 1, -1> > y_;
    };

    template <typename T1, typename T2>
    class vector_view_map<Eigen::Matrix<T1, 1, -1>, T2> {
      public:
        vector_view_map(const Eigen::Matrix<T1, 1, -1>& x, T2* y) 
         : y_(y) { }

        T2& operator[](int i) {
          return y_[i];
        }
      private:
        T2* y_;
    };

    template <typename T1, typename T2, int M, int N>
    class vector_view_map<Eigen::Matrix<T1, M, N>, Eigen::Matrix<T2, M, N> > {
      public:
        vector_view_map(const Eigen::Matrix<T1, M, N>& x, T2* y) 
         : y_(y, x.rows(), x.cols()) { }

        Eigen::Map<Eigen::Matrix<T2, M, N> >& operator[](int i) {
          return y_;
        }
      private:
        Eigen::Map<Eigen::Matrix<T2, M, N> > y_;
    };

    template <typename T1, typename T2, int M, int N>
    class vector_view_map<std::vector<Eigen::Matrix<T1, M, N> >,
                          Eigen::Matrix<T2, M, N> > {
      public:
        vector_view_map(const std::vector<Eigen::Matrix<T1, M, N> >& x, T2* y) 
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
