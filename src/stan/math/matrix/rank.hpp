#ifndef __STAN__MATH__MATRIX__RANK_HPP__
#define __STAN__MATH__MATRIX__RANK_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
   
    /**
     * Return the specified standard vector in ascending order.
     *
     * @param xs Standard vector to sum.
     * @return Standard vector ordered.
     * @tparam T Type of elements of the vector.
     */
    template <typename T>
    inline int rank(const std::vector<T> & v, int s)
    {
	s--;
	size_t size = v.size();
      if (s >= size)
	{
	  //FIXME issue warning
        return 0;
	}
	int count(0);
	T compare(v[s]);
      for (size_t i = 0; i < size; ++i)
	  if (v[i]<compare) count++;
      return count;
    }

    /**
     * Return the specified eigen vector in descending order.
     *
     * @param xs Eigen vector to sum.
     * @return Eigen vector ordered.
     * @tparam T Type of elements of the vector.
     */
 template <typename T, int R, int C>
    inline int rank(const Eigen::Matrix<T,R,C> & v, s)
    {
	s--;
	size_t size = v.size();
	const T * vv = v.data(); 
      if (s >= size)
	{
	  //FIXME issue warning	  
        return 0;
	}
	int count(0);
	T compare(vv[s]);
      for (size_t i = 0; i < size; ++i)
	  if (vv[i]<compare) count++;
      return count;
    }
    
  }
}
#endif
