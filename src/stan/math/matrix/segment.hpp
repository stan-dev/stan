#ifndef STAN__MATH__MATRIX__SEGMENT_HPP
#define STAN__MATH__MATRIX__SEGMENT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <vector>
#include <stan/error_handling/scalar/check_greater.hpp>
#include <stan/error_handling/scalar/check_less_or_equal.hpp>


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
      stan::error_handling::check_greater("segment", "n", i, 0.0);
      stan::error_handling::check_less_or_equal("segment", "n", i, static_cast<size_t>(v.rows()));
      if (n != 0) {
        stan::error_handling::check_greater("segment", "n", i+n-1, 0.0);
        stan::error_handling::check_less_or_equal("segment", "n", i+n-1, static_cast<size_t>(v.rows()));
      } 
      return v.segment(i-1,n);
    }

    template <typename T>
    inline
    Eigen::Matrix<T,1,Eigen::Dynamic>
    segment(const Eigen::Matrix<T,1,Eigen::Dynamic>& v,
            size_t i, size_t n) {
      stan::error_handling::check_greater("segment", "n", i, 0.0);
      stan::error_handling::check_less_or_equal("segment", "n", i, static_cast<size_t>(v.cols()));
      if (n != 0) {
        stan::error_handling::check_greater("segment", "n", i+n-1, 0.0);
        stan::error_handling::check_less_or_equal("segment", "n", i+n-1, static_cast<size_t>(v.cols()));
      } 
      
      return v.segment(i-1,n);
    }


    template <typename T>
    std::vector<T> 
    segment(const std::vector<T>& sv,
            size_t i, size_t n) {
      stan::error_handling::check_greater("segment", "i", i, 0.0);
      stan::error_handling::check_less_or_equal("segment", "i", i, sv.size());
      if (n != 0) {
        stan::error_handling::check_greater("segment", "i+n-1", i+n-1, 0.0);
        stan::error_handling::check_less_or_equal("segment", "i+n-1", i+n-1, static_cast<size_t>(sv.size()));
      }
      std::vector<T> s;
      for (size_t j = 0; j < n; ++j)
        s.push_back(sv[i + j - 1]);
      return s;
    }

  }
}
#endif
