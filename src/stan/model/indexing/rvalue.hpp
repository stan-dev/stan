#ifndef STAN_MODEL_INDEXING_RVALUE_HPP
#define STAN_MODEL_INDEXING_RVALUE_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <Eigen/Dense>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue_return.hpp>
#include <vector>

namespace stan {

  namespace model {

    /**
     * Return size of specified multi-index.
     *
     * @param[in] idx Input index.
     * @param[in] size Ignored.
     * @return Size of result structure.
     */
    inline int rvalue_index_size(const index_multi& idx, int size) {
      return idx.ns_.size();
    }

    /**
     * Return size of specified omni-index for specified size of
     * input. 
     *
     * @param[in] idx Input index.
     * @param[in] size Ignored.
     * @return Size of result structure.
     */
    inline int rvalue_index_size(const index_omni& idx, int size) {
      return size;
    }

    /**
     * Return size of specified min index for specified size of
     * input. 
     *
     * @param[in] idx Input index.
     * @param[in] size Ignored.
     * @return Size of result structure.
     */
    inline int rvalue_index_size(const index_min& idx, int size) {
      return size - idx.min_;
    }

    /**
     * Return size of specified max index.
     *
     * @param[in] idx Input index.
     * @param[in] size Ignored.
     * @return Size of result structure.
     */
    inline int rvalue_index_size(const index_max& idx, int size) {
      return idx.max_ + 1;
    }

    /**
     * Return size of specified min-max index.
     *
     * @param[in] idx Input index.
     * @param[in] size Ignored.
     * @return Size of result structure.
     */
    inline int rvalue_index_size(const index_min_max& idx, int size) {
      return idx.max_ - idx.min_ + 1;
    }


    /**
     * Return the index in the underlying array corresponding to the
     * specified position in the specified multi-index.
     
     * @param[in] n Relative index position.
     * @param[in] idx Index.
     * @return Underlying index position.
     */
    inline int rvalue_at(int n, const index_multi& idx) {
      return idx.ns_[n];
    }

    /**
     * Return the index in the underlying array corresponding to the
     * specified position in the specified omni-index.
     
     * @param[in] n Relative index position.
     * @param[in] idx Index.
     * @return Underlying index position.
     */
    inline int rvalue_at(int n, const index_omni& idx) {
      return n;
    }

    /**
     * Return the index in the underlying array corresponding to the
     * specified position in the specified min-index.
     
     * @param[in] n Relative index position.
     * @param[in] idx Index.
     * @return Underlying index position.
     */
    inline int rvalue_at(int n, const index_min& idx) {
      return idx.min_ + n;
    }

    /**
     * Return the index in the underlying array corresponding to the
     * specified position in the specified max-index.
     
     * @param[in] n Relative index position.
     * @param[in] idx Index.
     * @return Underlying index position.
     */
    inline int rvalue_at(int n, const index_max& idx) {
      return n;
    }

    /**
     * Return the index in the underlying array corresponding to the
     * specified position in the specified min-max-index.
     
     * @param[in] n Relative index position.
     * @param[in] idx Index.
     * @return Underlying index position.
     */
    inline int rvalue_at(int n, const index_min_max& idx) {
      return idx.min_ + n;
    }


    // T[] : T
    template <typename T>
    inline T rvalue(const T& c, const nil_index_list& /*idx*/) {
      return c;
    }

    // vec[single] : scal
    template <typename T>
    inline T rvalue(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v,
                    const cons_index_list<index_uni, nil_index_list>& idx) {
      return v(idx.head_.n_);
    }

    // rowvec[single] : scal
    template <typename T>
    inline T rvalue(const Eigen::Matrix<T, 1, Eigen::Dynamic>& v,
                    const cons_index_list<index_uni, nil_index_list>& idx) {
      return v(idx.head_.n_);
    }

    // vec[multiple] : vec
    template <typename T, typename I>
    inline
    typename boost::disable_if<boost::is_same<I, index_uni>,
                               Eigen::Matrix<T, Eigen::Dynamic, 1> >::type
    rvalue(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v,
           const cons_index_list<I, nil_index_list>& idx) {
      int size = rvalue_index_size(idx.head_, v.size());
      Eigen::Matrix<T, Eigen::Dynamic, 1> a(size);
      for (int i = 0; i < size; ++i)
        a(i) = v(rvalue_at(i, idx.head_));
      return a;
    }

    // rowvec[multiple] : rowvec
    template <typename T, typename I>
    inline
    typename boost::disable_if<boost::is_same<I, index_uni>,
                               Eigen::Matrix<T, 1, Eigen::Dynamic> >::type
    rvalue(const Eigen::Matrix<T, 1, Eigen::Dynamic>& v,
           const cons_index_list<I, nil_index_list>& idx) {
      int size = rvalue_index_size(idx.head_, v.size());
      Eigen::Matrix<T, 1, Eigen::Dynamic> a(size);
      for (int i = 0; i < size; ++i)
        a(i) = v(rvalue_at(i, idx.head_));
      return a;
    }

    // mat[single] : rowvec
    template <typename T>
    inline Eigen::Matrix<T, 1, Eigen::Dynamic>
    rvalue(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
          const cons_index_list<index_uni, nil_index_list>& idx) {
      return m.row(idx.head_.n_);
    }

    // mat[multiple] : mat
    template <typename T, typename I>
    inline typename boost::disable_if<boost::is_same<I, index_uni>,
                                      Eigen::Matrix<T, Eigen::Dynamic,
                                                    Eigen::Dynamic> >::type
    rvalue(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
           const cons_index_list<I, nil_index_list>& idx) {
      int n_rows = rvalue_index_size(idx.head_, m.rows());
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> a(n_rows, m.cols());
      for (int i = 0; i < n_rows; ++i)
        a.row(i) = m.row(rvalue_at(i, idx.head_));
      return a;
    }

    // mat[single,single] : scalar
    template <typename T>
    inline T
    rvalue(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
      const cons_index_list<index_uni,
                            cons_index_list<index_uni, nil_index_list> >& idx) {
      return m(idx.head_.n_, idx.tail_.head_.n_);
    }

    // mat[single,multiple] : row vector
    template <typename T, typename I>
    inline typename boost::disable_if<boost::is_same<I, index_uni>,
                                      Eigen::Matrix<T,
                                                    1, Eigen::Dynamic> >::type
    rvalue(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
           const cons_index_list<index_uni,
                                 cons_index_list<I, nil_index_list> >& idx) {
      Eigen::Matrix<T, 1, Eigen::Dynamic> r = m.row(idx.head_.n_);
      return rvalue(r, idx.tail_);
    }

    // mat[multiple,single] : vector
    template <typename T, typename I>
    inline
    typename boost::disable_if<boost::is_same<I, index_uni>,
                               Eigen::Matrix<T, Eigen::Dynamic, 1> >::type
    rvalue(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
       const
       cons_index_list<I, cons_index_list<index_uni, nil_index_list> >& idx) {
      int rows = rvalue_index_size(idx.head_, m.rows());
      Eigen::Matrix<T, Eigen::Dynamic, 1> c(rows);
      for (int i = 0; i < rows; ++i)
        c(i) = m(rvalue_at(i, idx.head_), idx.tail_.head_.n_);
      return c;
    }

    // mat[multiple,multiple] : mat
    template <typename T, typename I1, typename I2>
    inline
    typename boost::disable_if_c<boost::is_same<I1, index_uni>::value
                                 || boost::is_same<I2, index_uni>::value,
                                 Eigen::Matrix<T, Eigen::Dynamic,
                                               Eigen::Dynamic> >::type
    rvalue(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
         const cons_index_list<I1, cons_index_list<I2, nil_index_list> >& idx) {
      int rows = rvalue_index_size(idx.head_, m.rows());
      int cols = rvalue_index_size(idx.tail_.head_, m.cols());
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> c(rows, cols);
      for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
          c(i, j) = m(rvalue_at(i, idx.head_), rvalue_at(j, idx.tail_.head_));
      return c;
    }

    // std::vector<T>[single | L] : T[L]
    template <typename T, typename L>
    inline typename rvalue_return<std::vector<T>,
                                  cons_index_list<index_uni, L> >::type
    rvalue(const std::vector<T>& c, const cons_index_list<index_uni, L>& idx) {
      return rvalue(c[idx.head_.n_], idx.tail_);
    }

    // std::vector<T>[multiple | L] : std::vector<T[L]>
    template <typename T, typename I, typename L>
    inline typename rvalue_return<std::vector<T>, cons_index_list<I, L> >::type
    rvalue(const std::vector<T>& c, const cons_index_list<I, L>& idx) {
      typename rvalue_return<std::vector<T>,
                             cons_index_list<I, L> >::type result;
      for (int n = 0; n < rvalue_index_size(idx.head_, c.size()); ++n)
        result.push_back(rvalue(c[rvalue_at(n, idx.head_)], idx.tail_));
      return result;
    }


  }
}
#endif
