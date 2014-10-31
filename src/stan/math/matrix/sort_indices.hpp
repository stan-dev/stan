#ifndef STAN__MATH__MATRIX__SORT_INDICES_HPP
#define STAN__MATH__MATRIX__SORT_INDICES_HPP


#include <vector>
#include <algorithm>    // std::sort
#include <iostream>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>

namespace stan {

  namespace math {
    
    /**
     * A comparator that works for any container type that has the
     * brackets operator.
     *
     * @tparam ascending true if sorting in ascending order
     * @tparam C container type
     */
    namespace {
      template <bool ascending, typename C>
      class index_comparator {
         const C& xs_;
      public:
        /**
         * Construct an index comparator holding a reference
         * to the specified container.
         *
         * @patam xs Container
         */
         index_comparator(const C& xs) : xs_(xs) { }

        /**
         * Return true if the value at the first index is sorted in
         * front of the value at the second index;  this will depend
         * on the template parameter <code>ascending</code>.
         *
         * @param i Index of first value for comparison
         * @param j Index of second value for comparison
         */
        bool operator()(int i, int j) const {
           if (ascending)
             return xs_[i-1] < xs_[j-1];
           else
             return xs_[i-1] > xs_[j-1];
         }
      };

    
      /**
       * Return an integer array of indices of the specified container
       * sorting the values in ascending or descending order based on
       * the value of the first template prameter.
       *
       * @tparam ascending true if sort is in ascending order
       * @tparam C type of container
       * @param xs Container to sort
       * @return sorted version of container
       */
      template <bool ascending, typename C>
      std::vector<int> sort_indices(const C& xs) {
        typedef typename index_type<C>::type idx_t;
        idx_t size = xs.size();
        std::vector<int> idxs;
        idxs.resize(size);
        for (idx_t i = 0; i < size; ++i)
          idxs[i] = i + 1;
        index_comparator<ascending,C> comparator(xs);
        std::sort(idxs.begin(), idxs.end(), comparator);
        return idxs;
      }
    
    }
    
    /**
     * Return a sorted copy of the argument container in ascending order.
     *
     * @tparam C type of container
     * @param xs Container to sort
     * @return sorted version of container
     */
    template <typename C>
    std::vector<int> sort_indices_asc(const C& xs) {
      return sort_indices<true>(xs);
    }

    /**
     * Return a sorted copy of the argument container in ascending order.
     *
     * @tparam C type of container
     * @param xs Container to sort
     * @return sorted version of container
     */
    template <typename C>
    std::vector<int> sort_indices_desc(const C& xs) {
      return sort_indices<false>(xs);
    }


  }
}
#endif
