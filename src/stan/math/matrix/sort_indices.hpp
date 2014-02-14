#ifndef __STAN__MATH__MATRIX__SORT_INDICES_HPP__
#define __STAN__MATH__MATRIX__SORT_INDICES_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <vector>
#include <algorithm>    // std::sort



namespace stan {
  namespace math {
    
    namespace {
      struct svector_asc {
        template <typename T> bool operator() (std::pair <T, int> i,
                       std::pair <T, int> j) {
                        if (i.first!=j.first)
                          return (i.first < j.first);
                        else
                          return (i.second < j.second);
        }
      } vector_asc;
      
      struct svector_desc {
        template <typename T> bool operator() (std::pair <T, int> i,
                       std::pair <T, int> j) {
                        if (i.first!=j.first)
                          return (i.first > j.first);
                        else
                          return (i.second > j.second);
        }
      } vector_desc;
    }

    template <typename T>
    inline std::vector<int> sort_indices_asc(std::vector<T> xs) {
      size_t size = xs.size();
      if (size < 2) {
        if (size == 1)
          return std::vector<int>(1, 1);
        else
          return std::vector<int>();
      }
      
      std::vector<std::pair <T, int> > vector_pairs(size);

      for (size_t i=0; i != size; i++)     
        vector_pairs[i] = std::pair <T, int>(xs[i], i+1);

      std::sort(vector_pairs.begin(), vector_pairs.end(), vector_asc);

      std::vector<int> results(size);
      for (size_t i=0; i != size; i++)
        results[i] = vector_pairs[i].second;
      
      return results;
    }

    template <typename T>
    inline std::vector<int> sort_indices_desc(std::vector<T> xs) {
      size_t size = xs.size();
      if (size < 2) {
        if (size == 1)
          return std::vector<int>(1, 1);
        else
          return std::vector<int>();
      }
      
      std::vector<std::pair <T, int> > vector_pairs(size);

      for (size_t i=0; i < size; i++)
        vector_pairs[i] = std::pair <T, int>(xs[i], i+1);

      std::sort(vector_pairs.begin(), vector_pairs.end(), vector_desc);

      std::vector<int> results(size);
      for (size_t i=0; i < size; i++)
        results[i] = vector_pairs[i].second;

      return results;
    }
    
    template <typename T, int R, int C>
    inline std::vector<int> sort_indices_asc(Eigen::Matrix<T, R, C> xs) {
      ptrdiff_t size = xs.size();
      if (size < 2) {
        if (size == 1)
          return std::vector<int>(1, 1);
        else
          return std::vector<int>();
      }
      
      std::vector<std::pair <T, int> > vector_pairs(size);

      T* data = xs.data();
      for (ptrdiff_t i = 0; i < size; i++)
        vector_pairs[i] = std::pair <T, int>(data[i], i+1);

      std::sort(vector_pairs.begin(), vector_pairs.end(), vector_asc);

      std::vector<int> results(size);
      for (size_t i = 0; i < size; i++)
        results[i] = vector_pairs[i].second;

      return results;
    }

    template <typename T, int R, int C>
    inline std::vector<int> sort_indices_desc(Eigen::Matrix<T, R, C> xs) {
      ptrdiff_t size = xs.size();
      if (size < 2) {
        if (size == 1)
          return std::vector<int>(1, 1);
        else
          return std::vector<int>();
      }
      
      std::vector<std::pair <T, int> > vector_pairs(size);

      T* data = xs.data();
      for (ptrdiff_t i = 0; i < size; i++)
        vector_pairs[i] = std::pair <T, int>(data[i], i+1);

      std::sort(vector_pairs.begin(), vector_pairs.end(), vector_desc);

      std::vector<int> results(size);
      for (size_t i=0; i < size; i++)
        results[i] = vector_pairs[i].second;

      return results;
    }

  }
}
#endif
