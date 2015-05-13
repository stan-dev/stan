#ifndef STAN_MODEL_INDEXING_INDEXED_TYPE_HPP
#define STAN_MODEL_INDEXING_INDEXED_TYPE_HPP

#include <vector>
#include <stan/model/indexing/typelist.hpp>
#include <Eigen/Dense>

namespace stan {
  namespace model {

    /**
     * Class used to denote an index consisting of a single integer.
     *
     * See <code>multi_index</code>.
     */
    struct uni_index {
    };


    /**
     * Class used to denote an index consisting of multiple integers.
     *
     * See <code>uni_index</code>.
     */
    struct multi_index {
    };


    /**
     * Primary template class for metaprogram to calculate type of 
     * the result of an object of a specified type being indexed
     * by single or multiple indexes in a type list.
     *
     * <p>This primary template class is not meant to be implemented.
     * See its specializations.
     *
     * @tparam S value being indexed.
     * @tparam T type list of indexing types.
     */
    template <typename S = dummy, typename T = dummy>
    struct indexed_type {
    };

    /**
     * Template class specialization to calculate the type of an
     * object with an empty indexing list.
     *
     * @tparam S value being indexed.
     */
    template <typename S>
    struct indexed_type<S,nil> {

      /**
       * Typedef for result, i.e., template parameter <code>S</code>.
       */
      typedef S type;

    };


   // MATRIX WITH ONE INDEX 

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                        cons<uni_index, nil> > {
      typedef Eigen::Matrix<T, 1, Eigen::Dynamic> type;
    };

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                        cons<multi_index, nil> > {
      typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> type;
    };


    // MATRIX WITH TWO INDEXES

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                        cons<uni_index, cons<uni_index, nil> > > {
      typedef T type;
    };

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                        cons<multi_index, cons<uni_index, nil> > > {
      typedef Eigen::Matrix<T, Eigen::Dynamic, 1> type;
    };

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                        cons<uni_index, cons<multi_index, nil> > > {
      typedef Eigen::Matrix<T, 1, Eigen::Dynamic> type;
    };

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                        cons<multi_index, cons<multi_index, nil> > > {
      typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> type;
    };

    // VECTOR

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, Eigen::Dynamic, 1>,
                        cons<uni_index,nil> > {
      typedef T type;
    };

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, Eigen::Dynamic, 1>,
                        cons<multi_index,nil> > {
      typedef Eigen::Matrix<T, Eigen::Dynamic, 1> type;
    };

    // ROW VECTOR

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, 1, Eigen::Dynamic>,
                        cons<uni_index,nil> > {
      typedef T type;
    };

    template <typename T>
    struct indexed_type<Eigen::Matrix<T, 1, Eigen::Dynamic>,
                        cons<multi_index,nil> > {
      typedef Eigen::Matrix<T, 1, Eigen::Dynamic> type;
    };


    /**
     * Template class specialization to calculate the type of a
     * standard vector with the specified indexing list, where the
     * first index is the class encoding a single integer.
     *
     * @tparam S type of entries in vector object being indexed.
     * @tparam T tail of the type list of indexing types, which beings
     * with <code>uni_index</code>.
     */
    template <typename S, typename T>
    struct indexed_type<std::vector<S>, 
                        cons<uni_index, T> > {

      /**
       * Typedef for result, which recursively calculates the type of
       * indexing an object of type <code>S</code> by an index type
       * list <code>T</code>.
       */
      typedef typename indexed_type<S,T>::type type;

    };

    /**
     * Template class specialization to calculate the type of a
     * standard vector with the specified indexing list, where the
     * first index is the class encoding a multiple integer index.
     *
     * @tparam S type of entries in vector object being indexed.
     * @tparam T tail of the type list of indexing types, which beings
     * with <code>multi_index</code>.
     */
    template <typename S, typename T>
    struct indexed_type<std::vector<S>, 
                        cons<multi_index, T> > {
      
      /**
       * Typedef for result, which is a standard vector with entry
       * type calculated as the return type of an object of type
       * <code>S</code> indexed by a list of type <code>T</code>.
       */
      typedef std::vector<typename indexed_type<S,T>::type> type;
    };

  }
}
#endif
