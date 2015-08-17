#ifndef STAN_MODEL_INDEXING_LVALUE_HPP
#define STAN_MODEL_INDEXING_LVALUE_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <Eigen/Dense>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue.hpp>
#include <stdexcept>
#include <vector>

// ***** DOCUMENT ALIASING ISSUE if RHS isn't copy ******

// given sizings and rvalue copy, only for index_multi()

namespace stan {
  namespace model {

    // x[] = y
    template <typename T, typename U>
    inline void assign(T& x,
                       const nil_index_list& /* idxs */,
                       const U& y) {
      x = y;
    }

    // vec[single] : scalar
    template <typename T, typename U>
    inline void assign(Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
                       const cons_index_list<index_uni, nil_index_list>& idxs,
                       const U& y) {
      int i = idxs.head_.n_;
      if (i < 0 || i >= x.size()) {
        throw std::out_of_range("uni index out of range");
      }
      x(i) = y;
    }

    // rowvec[single] : scalar
    template <typename T, typename U>
    inline void assign(Eigen::Matrix<T, 1, Eigen::Dynamic>& x,
                       const cons_index_list<index_uni, nil_index_list>& idxs,
                       const U& y) {
      int i = idxs.head_.n_;
      if (i < 0 || i >= x.size())
        throw std::out_of_range("uni index out of range");
      x(i) = y;
    }

    // vec[multiple] : vec
    template <typename T, typename I, typename U>
    inline typename boost::disable_if<boost::is_same<I, index_uni>, void>::type
    assign(Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
           const cons_index_list<I, nil_index_list>& idxs,
           const Eigen::Matrix<U, Eigen::Dynamic, 1>& y) {
      for (int n = 0; n < y.size(); ++n) {
        int i = rvalue_at(n, idxs.head_);
        if (i < 0 || i >= x.size())
          throw std::out_of_range("multi index out of range");
        x(i) = y(n);
      }
    }

    // rowvec[multiple] : rowvec
    template <typename T, typename I, typename U>
    inline typename boost::disable_if<boost::is_same<I, index_uni>, void>::type
    assign(Eigen::Matrix<T, 1, Eigen::Dynamic>& x,
           const cons_index_list<I, nil_index_list>& idxs,
           const Eigen::Matrix<U, 1, Eigen::Dynamic>& y) {
      for (int n = 0; n < y.size(); ++n) {
        int i = rvalue_at(n, idxs.head_);
        if (i < 0 || i >= x.size())
          throw std::out_of_range("multi index out of range");
        x(i) = y(n);
      }
    }

    // mat[single] = rowvec
    template <typename T>
    void assign(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& x,
                const cons_index_list<index_uni, nil_index_list>& idxs,
                const Eigen::Matrix<T, 1, Eigen::Dynamic>& y) {
      int i = idxs.head_.n_;
      if (i < 0 || i >= x.rows())
        throw std::out_of_range("matrix[uni] out of range");
      x.row(idxs.head_.n_) = y;
    }

    // mat[multiple] = mat
    template <typename T, typename I>
    inline typename boost::disable_if<boost::is_same<I, index_uni>, void>::type
    assign(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& x,
           const cons_index_list<I, nil_index_list>& idxs,
           const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& y) {
      for (int i = 0; i < y.rows(); ++i) {
        int m = rvalue_at(i, idxs.head_);
        if (m < 0 || m > x.rows())
          throw std::out_of_range("matrix[multi] out of range");
        x.row(m) = y.row(i);
      }
    }

    // mat[single, single] = scalar
    template <typename T, typename U>
    void assign(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& x,
                const cons_index_list<index_uni,
                                      cons_index_list<index_uni,
                                                      nil_index_list> >& idxs,
                const U& y) {
      x(idxs.head_.n_, idxs.tail_.head_.n_) = y;
    }

    // mat[single, multiple] = rowvec
    template <typename T, typename U, typename I>
    inline typename boost::disable_if<boost::is_same<I, index_uni>, void>::type
    assign(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& x,
           const cons_index_list<index_uni,
                                 cons_index_list<I, nil_index_list> >& idxs,
           const Eigen::Matrix<U, 1, Eigen::Dynamic>& y) {
      for (int i = 0; i < y.size(); ++i) {
        int j = rvalue_at(i, idxs.tail_.head_);
        x(idxs.head_.n_, j) = y(i);
      }
    }

    // mat[multiple, single] = vec
    template <typename T, typename U, typename I>
    inline typename boost::disable_if<boost::is_same<I, index_uni>, void>::type
    assign(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& x,
           const cons_index_list<I,
                                 cons_index_list<index_uni,
                                                 nil_index_list> >& idxs,
           const Eigen::Matrix<U, Eigen::Dynamic, 1>& y) {
      for (int i = 0; i < y.size(); ++i) {
        int m = rvalue_at(i, idxs.head_);
        x(m, idxs.tail_.head_.n_) = y(i);
      }
    }

    // mat[multiple, multiple] = mat
    template <typename T, typename U, typename I1, typename I2>
    inline typename
    boost::disable_if_c<boost::is_same<I1, index_uni>::value
                        || boost::is_same<I2, index_uni>::value, void>::type
    assign(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& x,
           const cons_index_list<I1,
                                 cons_index_list<I2, nil_index_list> >& idxs,
           const Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic>& y) {
      for (int j = 0; j < y.cols(); ++j) {
        int n = rvalue_at(j, idxs.tail_.head_);
        for (int i = 0; i < y.rows(); ++i) {
          int m = rvalue_at(i, idxs.head_);
          x(m, n) = y(i, j);
        }
      }
    }

    // x[single | L] = y
    template <typename T, typename L, typename U>
    inline void assign(std::vector<T>& x,
                       const cons_index_list<index_uni, L>& idxs,
                       const U& y) {
      if (idxs.head_.n_ < 0 || idxs.head_.n_ >= static_cast<int>(x.size())) {
        throw std::out_of_range("uni index out of range");
      }
      assign(x[idxs.head_.n_], idxs.tail_, y);
    }

    // x[multiple | L] = y
    template <typename T, typename I, typename L, typename U>
    typename boost::disable_if<boost::is_same<I, index_uni>, void>::type
    inline assign(std::vector<T>& x,
                  const cons_index_list<I, L>& idxs,
                  const std::vector<U>& y) {
      for (size_t n = 0; n < y.size(); ++n) {
        int i = rvalue_at(n, idxs.head_);
        if (i < 0 || i >= static_cast<int>(x.size())) {
          throw std::out_of_range("index out of range");
        }
        assign(x[i], idxs.tail_, y[n]);
      }
    }



  }
}
#endif
