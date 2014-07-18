#ifndef STAN__MATH__MATRIX__SQUARED_DISTANCE_HPP
#define STAN__MATH__MATRIX__SQUARED_DISTANCE_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_vector.hpp>
#include <stan/math/error_handling/matrix/check_matching_sizes.hpp>

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
    inline double squared_distance(const Eigen::Matrix<double, R1, C1>& v1, 
                                   const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::check_vector("squared_distance(%1%)",v1,"v1",(double*)0);
      stan::math::check_vector("squared_distance(%1%)",v2,"v2",(double*)0);
      stan::math::check_matching_sizes("squared_distance(%1%)",v1,"v1",
                                       v2,"v2",(double*)0);
      if (v1.rows() != v2.rows())
        return (v1.transpose()-v2).squaredNorm();
      else
        return (v1-v2).squaredNorm();
    }
  }
}
#endif
