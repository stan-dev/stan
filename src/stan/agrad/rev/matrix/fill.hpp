#ifndef __STAN__AGRAD__REV__MATRIX__FILL_HPP__
#define __STAN__AGRAD__REV__MATRIX__FILL_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace agrad {

    template <typename T, typename S>
    void fill(T& x, const S& y) {
      x = y;
    }
    
    template <typename T, int R, int C, typename S>
    void fill(Eigen::Matrix<T,R,C>& x, const S& y) {
      x.fill(y);
    }

    template <typename T, typename S>
    void fill(std::vector<T>& x, const S& y) {
      for (size_t i = 0; i < x.size(); ++i)
        fill(x[i],y);
    }

  }
}
#endif
