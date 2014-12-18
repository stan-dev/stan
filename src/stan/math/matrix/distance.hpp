#ifndef STAN__MATH__MATRIX__DISTANCE_HPP
#define STAN__MATH__MATRIX__DISTANCE_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/squared_distance.hpp>
#include <stan/error_handling/matrix/check_vector.hpp>
#include <stan/error_handling/matrix/check_matching_sizes.hpp>

namespace stan {
  namespace math {
    
    /**
     * Returns the distance between the specified vectors.
     *
     * @param v1 First vector.
     * @param v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error If the vectors are not the same
     * size or if they are both not vector dimensioned.
     */
    template<typename T1, int R1,int C1, typename T2, int R2, int C2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    distance(const Eigen::Matrix<T1, R1, C1>& v1,
             const Eigen::Matrix<T2, R2, C2>& v2) {
      using std::sqrt;
      stan::error_handling::check_vector("distance", "v1", v1);
      stan::error_handling::check_vector("distance", "v2", v2);
      stan::error_handling::check_matching_sizes("distance",
                                                 "v1", v1,
                                                 "v2", v2);
      return sqrt(squared_distance(v1,v2));
    }
  }
}
#endif
