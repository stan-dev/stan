#ifndef __STAN__AGRAD__REV__SORT_INDICES_HPP__
#define __STAN__AGRAD__REV__SORT_INDICES_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <algorithm>    // std::sort

namespace stan {
  namespace agrad {

    namespace {
      struct svector_asc_rev {
        bool operator() (std::pair <double, int> i,
                         std::pair <double, int> j) {
                        if (i.first!=j.first)
                          return (i.first < j.first);
                        else
                          return (i.second < j.second);
        }
      } vector_asc_rev;
      
      struct svector_desc_rev {
        bool operator() (std::pair <double, int> i,
                         std::pair <double, int> j) {
                        if (i.first!=j.first)
                          return (i.first > j.first);
                        else
                          return (i.second > j.second);
        }
      } vector_desc_rev;
    }

    inline std::vector<int> sort_indices_asc(std::vector<var> xs) {
      size_t size = xs.size();
      if (size < 2) {
        if (size == 1)
          return std::vector<int>(1, 1);
        else
          return std::vector<int>();
      }
      
      std::vector<std::pair <double, int> > vector_pairs(size);

      for (size_t i=0; i != size; i++)     
        vector_pairs[i] = std::pair <double, int>(xs[i].val(), i+1);

      std::sort(vector_pairs.begin(), vector_pairs.end(), vector_asc_rev);

      std::vector<int> results(size);
      for (size_t i=0; i != size; i++)
        results[i] = vector_pairs[i].second;
      
      return results;
    }

    inline std::vector<int> sort_indices_desc(std::vector<var> xs) {
      size_t size = xs.size();
      if (size < 2) {
        if (size == 1)
          return std::vector<int>(1, 1);
        else
          return std::vector<int>();
      }
      
      std::vector<std::pair <double, int> > vector_pairs(size);

      for (size_t i=0; i < size; i++)
        vector_pairs[i] = std::pair <double, int>(xs[i].val(), i+1);

      std::sort(vector_pairs.begin(), vector_pairs.end(), vector_desc_rev);

      std::vector<int> results(size);
      for (size_t i=0; i < size; i++)
        results[i] = vector_pairs[i].second;

      return results;
    }
    
    template <int R, int C>
    inline std::vector<int> sort_indices_asc(Eigen::Matrix<var, R, C> xs) {
      ptrdiff_t size = xs.size();
      if (size < 2) {
        if (size == 1)
          return std::vector<int>(1, 1);
        else
          return std::vector<int>();
      }
      
      std::vector<std::pair <double, int> > vector_pairs(size);

      var* data = xs.data();
      for (ptrdiff_t i = 0; i < size; i++)
        vector_pairs[i] = std::pair <double, int>(data[i].val(), i+1);;

      std::sort(vector_pairs.begin(), vector_pairs.end(), vector_asc_rev);

      std::vector<int> results(size);
      for (size_t i=0; i < size; i++)
        results[i] = vector_pairs[i].second;

      return results;
    }

    template <int R, int C>
    inline std::vector<int> sort_indices_desc(Eigen::Matrix<var, R, C> xs) {
      ptrdiff_t size = xs.size();
      if (size < 2) {
        if (size == 1)
          return std::vector<int>(1, 1);
        else
          return std::vector<int>();
      }
      
      std::vector<std::pair <double, int> > vector_pairs(size);

      var* data = xs.data();
      for (ptrdiff_t i = 0; i < size; i++)
        vector_pairs[i] = std::pair <double, int>(data[i].val(), i+1);

      std::sort(vector_pairs.begin(), vector_pairs.end(), vector_desc_rev);

      std::vector<int> results(size);
      for (size_t i=0; i < size; i++)
        results[i] = vector_pairs[i].second;

      return results;
    }

  }
}
#endif
