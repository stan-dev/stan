#ifndef STAN__MATH__MATRIX__NUM_ELEMENTS_HPP
#define STAN__MATH__MATRIX__NUM_ELEMENTS_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    namespace {
          
      template <typename T>
      inline int
      num_elements_helper(const std::vector<T>& x, int n) {
        return x.size()*n;
      }
      
      template <typename T, int R, int C>
      inline int
      num_elements_helper(const std::vector<Eigen::Matrix<T,R,C> >& x, int n) {
        size_t size_ = x.size();
        if (size_ == 0)
          return 0;
        else  
          return x[0].size() * size_ * n;
      }

      template <typename T>
      inline int
      num_elements_helper(const std::vector<std::vector<T> >& x, int n) {
        size_t size_ = x.size();
        if (size_ == 0)
          return 0;
        else  
          return num_elements_helper(x[0], size_*n);
      }
    
    }
    
    template <typename T>
    inline int
    num_elements(const std::vector<T>& x) {
      return num_elements_helper(x, 1);
    }

    template <typename T, int R, int C>
    inline int 
    num_elements(const Eigen::Matrix<T,R,C>& m) {
      return m.size();
    }

  }
}
#endif
