#ifndef STAN__MATH__MATRIX__SQUARED_DISTANCE_HPP
#define STAN__MATH__MATRIX__SQUARED_DISTANCE_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_vector.hpp>
#include <stan/error_handling/matrix/check_matching_sizes.hpp>

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
    template<int R1,int C1,int R2, int C2, typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    squared_distance(const Eigen::Matrix<T1, R1, C1>& v1, 
                     const Eigen::Matrix<T2, R2, C2>& v2) {
      stan::error_handling::check_vector("squared_distance", "v1", v1);
      stan::error_handling::check_vector("squared_distance", "v2", v2);
      stan::error_handling::check_matching_sizes("squared_distance", 
                                                 "v1", v1,
                                                 "v2", v2);
      if (v1.rows() != v2.rows())
        return (v1.transpose()-v2).squaredNorm();
      else
        return (v1-v2).squaredNorm();
    }
  }
}
#endif
