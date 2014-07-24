#ifndef STAN__MATH__MATRIX__SEGMENT_HPP
#define STAN__MATH__MATRIX__SEGMENT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <vector>
#include <stan/math/error_handling/check_greater.hpp>
#include <stan/math/error_handling/check_less_or_equal.hpp>


namespace stan {
  namespace math {

    /**
     * Return the specified number of elements as a vector starting
     * from the specified element - 1 of the specified vector.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    segment(const Eigen::Matrix<T,Eigen::Dynamic,1>& v,
            size_t i, size_t n) {
      stan::math::check_greater("segment(%1%)",i,0.0,"n",(double*)0);
      stan::math::check_less_or_equal("segment(%1%)",i,static_cast<size_t>(v.rows()),"n",(double*)0);
      if (n != 0) {
        stan::math::check_greater("segment(%1%)",i+n-1,0.0,"n",(double*)0);
        stan::math::check_less_or_equal("segment(%1%)",i+n-1,static_cast<size_t>(v.rows()),"n",(double*)0);
      } 
      return v.segment(i-1,n);
    }

    template <typename T>
    inline
    Eigen::Matrix<T,1,Eigen::Dynamic>
    segment(const Eigen::Matrix<T,1,Eigen::Dynamic>& v,
            size_t i, size_t n) {
      stan::math::check_greater("segment(%1%)",i,0.0,"n",(double*)0);
      stan::math::check_less_or_equal("segment(%1%)",i,static_cast<size_t>(v.cols()),"n",(double*)0);    
      if (n != 0) {
        stan::math::check_greater("segment(%1%)",i+n-1,0.0,"n",(double*)0);
        stan::math::check_less_or_equal("segment(%1%)",i+n-1,static_cast<size_t>(v.cols()),"n",(double*)0);
      } 
      
      return v.segment(i-1,n);
    }


    template <typename T>
    std::vector<T> 
    segment(const std::vector<T>& sv,
            size_t i, size_t n) {
      stan::math::check_greater("segment(%1%)",i,0.0,"i",(double*)0);
      stan::math::check_less_or_equal("segment(%1%)",i,sv.size(),"i",(double*)0);
      if (n != 0) {
        stan::math::check_greater("segment(%1%)",i+n-1,0.0,"i+n-1",(double*)0);
        stan::math::check_less_or_equal("segment(%1%)",i+n-1,static_cast<size_t>(sv.size()),"i+n-1",
                                        (double*)0);
      }
      std::vector<T> s;
      for (size_t j = 0; j < n; ++j)
        s.push_back(sv[i + j - 1]);
      return s;
    }

  }
}
#endif
