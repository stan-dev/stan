#ifndef STAN_MODEL_INDEXING_RVALUE_RETURN_HPP
#define STAN_MODEL_INDEXING_RVALUE_RETURN_HPP

#include <vector>
#include <Eigen/Dense>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>

namespace stan {
  namespace model {

    // primary struct for metaprogram to calculate return
    // type for rvalue()

    template <typename C, typename L>
    struct rvalue_return {
    };

    // NO INDEX

    // C[] : C
    template <typename C>
    struct rvalue_return<C, nil_index_list> {
      typedef C type;
    };

    // SINGLE INDEX

    // matrix[multi] : matrix
    // vector[multi] : vector
    // row_vector[multi] : row_vector
    template <typename T, typename I, int R, int C>
    struct rvalue_return<Eigen::Matrix<T, R, C>, 
                         cons_index_list<I, nil_index_list> > {
      typedef Eigen::Matrix<T, R, C> type;
    };

    // matrix[uni] : row_vector
    template <typename T>
    struct rvalue_return<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 
                         cons_index_list<index_uni, nil_index_list> > {
      typedef Eigen::Matrix<T, 1, Eigen::Dynamic> type;
    };

    // vector[uni] : scalar
    template <typename T>
    struct rvalue_return<Eigen::Matrix<T, Eigen::Dynamic, 1>, 
                         cons_index_list<index_uni, nil_index_list> > {
      typedef T type;
    };

    // row_vector[uni] : scalar
    template <typename T>
    struct rvalue_return<Eigen::Matrix<T, 1, Eigen::Dynamic>, 
                         cons_index_list<index_uni, nil_index_list> > {
      typedef T type;
    };

    // TWO INDEXES

    // matrix[multi, multi] : matrix
    template <typename T, typename I1, typename I2>
    struct rvalue_return<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 
                         cons_index_list<I1, 
                                         cons_index_list<I2,
                                                         nil_index_list> > > {
      typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> type;
    };

    // matrix[multi, uni] : vector
    template <typename T, typename I>
    struct rvalue_return<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 
                         cons_index_list<I, 
                                         cons_index_list<index_uni,
                                                         nil_index_list> > > {
      typedef Eigen::Matrix<T, Eigen::Dynamic, 1> type;
    };

    // matrix[uni, multi] : row_vector
    template <typename T, typename I>
    struct rvalue_return<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 
                         cons_index_list<index_uni, 
                                         cons_index_list<I,
                                                         nil_index_list> > > {
      typedef Eigen::Matrix<T, 1, Eigen::Dynamic> type;
    };

    // matrix[uni, uni] : scalar
    template <typename T>
    struct rvalue_return<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 
                         cons_index_list<index_uni, 
                                         cons_index_list<index_uni,
                                                         nil_index_list> > > {
      typedef T type;
    };


    // STD VECTOR
    
    // std::vector<C>[multi | L] : std::vector<typeof(C[L])>
    template <typename C, typename I, typename L>
    struct rvalue_return<std::vector<C>, cons_index_list<I, L> > {
      typedef std::vector<typename rvalue_return<C, L>::type> type;
    };

    // std::vector<C>[uni | L] : typeof(C[L])
    template <typename C, typename L>
    struct rvalue_return<std::vector<C>, cons_index_list<index_uni, L> > {
      typedef typename rvalue_return<C, L>::type type;
    };

  }
}
#endif
