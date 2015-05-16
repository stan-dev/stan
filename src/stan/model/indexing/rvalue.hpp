#ifndef STAN_MODEL_INDEXING_RVALUE_HPP
#define STAN_MODEL_INDEXING_RVALUE_HPP

#include <vector>
#include <Eigen/Dense>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue_return.hpp>

// ******** ALL OFF BY ONE FOR LANGUAGE **********

namespace stan {
  namespace model {


    inline int rvalue_index_size(const index_multi& idx, int size) {
      return idx.ns_.size();
    }
    inline int rvalue_index_size(const index_omni& idx, int size) {
      return size;
    }
    inline int rvalue_index_size(const index_min& idx, int size) {
      return size - idx.min_;
    }
    inline int rvalue_index_size(const index_max& idx, int size) {
      return idx.max_ + 1;
    }
    inline int rvalue_index_size(const index_min_max& idx, int size) {
      return idx.max_ - idx.min_ + 1;
    }

    inline int rvalue_at(int n, const index_multi& idx) {
      return idx.ns_[n];
    }
    inline int rvalue_at(int n, const index_omni& idx) {
      return n;
    }
    inline int rvalue_at(int n, const index_min& idx) {
      return idx.min_ + n;
    }
    inline int rvalue_at(int n, const index_max& idx) {
      return n;
    }
    inline int rvalue_at(int n, const index_min_max& idx) {
      return idx.min_ + n;
    }


    typedef cons_index_list<index_uni, nil_index_list> single_index_list_t;

    typedef cons_index_list<index_uni, single_index_list_t> 
    single_single_index_list_t;

    /**
     * Primary template structure for the rvalue indexer.
     * Specializations will implement a static function
     * <code>apply()</code> function mapping a container of type
     * <code>C</code> and index list of type <code>I</code> to the
     * result of applying the indexing.
     *
     * @tparam C type of container.
     * @tparam I index type list.
     */
    template <typename C, typename I>
    struct rvalue_indexer {
    };


    // C[]
    template <typename C>
    struct rvalue_indexer<C, nil_index_list> {
      static inline C apply(const C& c, const nil_index_list& /*idx*/) {
        return c;
      }
    };
    
    // std::vector<T>[single | L]
    template <typename C, typename L>
    struct rvalue_indexer<C, cons_index_list<index_uni, L> > {
      typedef cons_index_list<index_uni, L> index_t;

      typedef typename rvalue_return<C, index_t>::type return_t;

      static inline return_t apply(const C& c, const index_t& idx) {
        return rvalue(c[idx.head_.n_], idx.tail_);
      }
    };

    // std::vector<T>[multiple | L]
    template <typename C, typename I, typename L>
    struct rvalue_indexer<C, cons_index_list<I, L> > {
      typedef cons_index_list<I, L> index_t;

      typedef typename rvalue_return<C, index_t>::type return_t;
      
      static inline return_t apply(const C& c, const index_t& idx) {
        return_t result;
        for (int n = 0; n < rvalue_index_size(idx.head_, c.size()); ++n)
          result.push_back(rvalue(c[rvalue_at(n, idx.head_)], idx.tail_));
        return result;
      }
    };

    // vec[single]
    template <typename T>
    struct rvalue_indexer<Eigen::Matrix<T, Eigen::Dynamic, 1>,
                          single_index_list_t> {
      static inline T 
      apply(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v,
            const single_index_list_t& idx) {
        return v(idx.head_.n_);
      }
    };

    // vec[multiple]
    template <typename T, typename I>
    struct rvalue_indexer<Eigen::Matrix<T, Eigen::Dynamic, 1>,
                          cons_index_list<I, nil_index_list> > {
      static inline Eigen::Matrix<T, Eigen::Dynamic, 1> 
      apply(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v,
            const cons_index_list<I, nil_index_list>& idx) {
        int size = rvalue_index_size(idx.head_, v.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> a(size);
        for (int i = 0; i < size; ++i)
          a(i) = v(rvalue_at(i, idx.head_));
        return a;
      }
    };

    // rowvec[single]
    template <typename T>
    struct rvalue_indexer<Eigen::Matrix<T, 1, Eigen::Dynamic>,
                          single_index_list_t> {
      static inline T 
      apply(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v,
            const single_index_list_t& idx) {
        return v(idx.head_.n_);
      }
    };

    // rowvec[multiple]
    template <typename T, typename I>
    struct rvalue_indexer<Eigen::Matrix<T, 1, Eigen::Dynamic>,
                          cons_index_list<I, nil_index_list> > {
      static inline Eigen::Matrix<T, 1, Eigen::Dynamic> 
      apply(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v,
            const cons_index_list<I, nil_index_list>& idx) {
        int size = rvalue_index_size(idx.head_, v.size());
        Eigen::Matrix<T, 1, Eigen::Dynamic> a(size);
        for (int i = 0; i < size; ++i)
          a(i) = v(rvalue_at(i, idx.head_));
        return a;
      }
    };


    // mat[single] : rowvec
    template <typename T>
    struct rvalue_indexer<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                          single_index_list_t> {
      static inline Eigen::Matrix<T, 1, Eigen::Dynamic> 
      apply(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
            const cons_index_list<index_uni, nil_index_list>& idx) {
        return m.row(idx.head_.n_);
      }
    };

    // mat[multiple] : mat
    template <typename T, typename I>
    struct rvalue_indexer<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                          cons_index_list<I, nil_index_list> > {
      static inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> 
      apply(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
            const cons_index_list<I, nil_index_list>& idx) {
        int n_rows = rvalue_index_size(idx.head_, m.rows());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> a(n_rows, m.cols());
        for (int i = 0; i < n_rows; ++i)
          a.row(i) = m.row(rvalue_at(i, idx.head_));
        return a;
      }
    };

    // mat[single,single] : scalar
    template <typename T> 
    struct rvalue_indexer<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                          single_single_index_list_t> {
      static inline T 
      apply(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
            const single_single_index_list_t& idx) {
        return m(idx.head_.n_, idx.tail_.head_.n_);
      }
    };







    // use recursion and done!
    // mat[single,multiple] : row vector
    // mat[multiple,single] : vector
    // mat[multiple,multiple] : matrix


    /**
     * Return the result of indexing the specified container
     * with the specified index list.
     *
     * <p>The return type reduces dimensions where the index
     * provides a single index.
     *
     * @tparam C type of container.
     * @tparam I type of index list.
     * @param[in] c container.
     * @param[in] idx index.
     * @return slice of container picked out by index.
     */
    template <typename C, typename I>
    inline typename rvalue_return<C, I>::type 
    rvalue(const C& c, const I& idx) {
      return rvalue_indexer<C, I>::apply(c, idx);
    }

  }
}
#endif

