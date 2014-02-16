#ifndef __STAN__MATH__MATRIX__SORT_INDICES_HPP__
#define __STAN__MATH__MATRIX__SORT_INDICES_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <vector>
#include <algorithm>    // std::sort
#include <iostream>

namespace stan {
  namespace math {
    
    namespace {
      template <bool ascending, typename T>
      class index_comparator_stdvector {
         const std::vector<T>& xs_;
      public:
         index_comparator_stdvector(const std::vector<T>& xs) : xs_(xs) { }
         bool operator()(int i, int j) const {
           if (ascending)
             return xs_[i-1] < xs_[j-1];
           else
             return xs_[i-1] > xs_[j-1];
         }
      };

    
      template <bool ascending, typename T>
      std::vector<int> sort_indices(const std::vector<T>& xs) {
        size_t size = xs.size();
        std::vector<int> idxs(size);
        for (int i = 0; i < size; ++i)
          idxs[i] = i + 1;
        index_comparator_stdvector<ascending,T> comparator(xs);
        std::sort(idxs.begin(), idxs.end(),
                  comparator);
        return idxs;
      }
    
    }
    
    template <typename T>
    std::vector<int> sort_indices_asc(const std::vector<T>& xs) {
      return sort_indices<true>(xs);
    }

    template <typename T>
    std::vector<int> sort_indices_desc(const std::vector<T>& xs) {
      return sort_indices<false>(xs);
    }
    
    
    //Same as all above, but for eigen matrices
    namespace {
      template <bool ascending, int R, int C, typename T>
      class index_comparator_eigen {
         const Eigen::Matrix<T, R, C>& xs_;
      public:
         index_comparator_eigen(const Eigen::Matrix<T, R, C>& xs) : xs_(xs) { }
         bool operator()(int i, int j) const {
           if (ascending)
             return xs_[i-1] < xs_[j-1];
           else
             return xs_[i-1] > xs_[j-1];
         }
      };

    
      template <bool ascending, int R, int C, typename T>
      std::vector<int> sort_indices(const Eigen::Matrix<T, R, C>& xs) {
        size_t size = xs.size();
        std::vector<int> idxs(size);
        for (int i = 0; i < size; ++i)
          idxs[i] = i + 1;
        index_comparator_eigen<ascending, R, C, T> comparator(xs);
        std::sort(idxs.begin(), idxs.end(),
                  comparator);
        return idxs;
      }
    
    }
    
    template <typename T, int R, int C>
    std::vector<int> sort_indices_asc(const Eigen::Matrix<T, R, C>& xs) {
      return sort_indices<true, R, C>(xs);
    }

    template <typename T, int R, int C>
    std::vector<int> sort_indices_desc(const Eigen::Matrix<T, R, C>& xs) {
      return sort_indices<false, R, C>(xs);
    }

  }
}
#endif
