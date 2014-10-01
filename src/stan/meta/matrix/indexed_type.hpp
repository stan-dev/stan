#ifndef STAN__META__MATRIX__INDEXED_TYPE_HPP
#define STAN__META__MATRIX__INDEXED_TYPE_HPP

#include <vector>

#include <stan/math/matrix/Eigen.hpp>

#include <stan/meta/indexed_type.hpp>
#include <stan/meta/typelist.hpp>

namespace stan {

  namespace meta {

    // Eigen vector

    template <typename S>
    struct indexed_type<Eigen::Matrix<S,Eigen::Dynamic,1>, 
                        cons<uni_index, nil> > {
      typedef S type;
    };

    template <typename S>
    struct indexed_type<Eigen::Matrix<S,Eigen::Dynamic,1>, 
                        cons<multi_index, nil> > {
      typedef Eigen::Matrix<S,Eigen::Dynamic,1> type;
    };



    // Eigen row vector

    template <typename S>
    struct indexed_type<Eigen::Matrix<S,1,Eigen::Dynamic>, 
                        cons<uni_index, nil> > {
      typedef S type;
    };

    template <typename S>
    struct indexed_type<Eigen::Matrix<S,1,Eigen::Dynamic>, 
                        cons<multi_index, nil> > {
      typedef Eigen::Matrix<S,1,Eigen::Dynamic> type;
    };



    // Eigen matrix

    template <typename S>
    struct indexed_type<Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic>,
                        cons<uni_index, nil> > {
      typedef Eigen::Matrix<S,1,Eigen::Dynamic> type;
    };


    template <typename S>
    struct indexed_type<Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic>,
                        cons<multi_index, nil> > {
      typedef Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic> type;
    };


    template <typename S>
    struct indexed_type<Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic>,
                        cons<uni_index, cons<uni_index, nil> > > {
      typedef double type;
    };

    template <typename S>
    struct indexed_type<Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic>,
                        cons<uni_index, cons<multi_index, nil> > > {
      typedef Eigen::Matrix<S,1,Eigen::Dynamic> type;
    };

    template <typename S>
    struct indexed_type<Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic>,
                        cons<multi_index, cons<uni_index, nil> > > {
      typedef Eigen::Matrix<S,Eigen::Dynamic,1> type;
    };

    template <typename S>
    struct indexed_type<Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic>,
                        cons<multi_index, cons<multi_index, nil> > > {
      typedef Eigen::Matrix<S,Eigen::Dynamic,Eigen::Dynamic> type;
    };


  }
}

#endif
