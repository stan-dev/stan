#ifndef __STAN__MATH__MATRIX__DIST_HPP__
#define __STAN__MATH__MATRIX__DIST_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/squared_dist.hpp>

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
    inline typename boost::math::tools::promote_args<T1,T1>::type
    dist(const Eigen::Matrix<T1, R1, C1>& v1,
         const Eigen::Matrix<T2, R2, C2>& v2) {
      using std::sqrt;
      return sqrt(squared_dist(v1,v2));
    }
  }
}
#endif
