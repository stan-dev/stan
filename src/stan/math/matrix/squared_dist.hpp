#ifndef __STAN__MATH__MATRIX__SQUARED_DIST_HPP__
#define __STAN__MATH__MATRIX__SQUARED_DIST_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_vector.hpp>
#include <stan/math/matrix/validate_matching_sizes.hpp>

namespace stan {
  namespace math {
    
    /**
     * Returns the squared distance between the specified vectors.
     *
     * @param v1 First vector.
     * @param v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error If the vectors are not the same
     * size or if they are both not vector dimensioned.
     */
    template<int R1,int C1,int R2, int C2>
    inline double squared_dist(const Eigen::Matrix<double, R1, C1>& v1, 
                               const Eigen::Matrix<double, R2, C2>& v2) {
      validate_vector(v1,"squared_dist");
      validate_vector(v2,"squared_dist");
      validate_matching_sizes(v1,v2,"squared_dist");
      if (v1.rows() != v2.rows())
        return (v1.transpose()-v2).squaredNorm();
      else
        return (v1-v2).squaredNorm();
    }
  }
}
#endif
