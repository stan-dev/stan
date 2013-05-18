#ifndef __STAN__MATH__MATRIX__SEGMENT_HPP__
#define __STAN__MATH__MATRIX__SEGMENT_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_row_index.hpp>
#include <stan/math/matrix/validate_column_index.hpp>
#include <stan/math/matrix/validate_std_vector_index.hpp>


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
      validate_row_index(v, i, "segment");
      if (n != 0) validate_row_index(v, i + n - 1, "segment");
      return v.segment(i-1,n);
    }

    template <typename T>
    inline
    Eigen::Matrix<T,1,Eigen::Dynamic>
    segment(const Eigen::Matrix<T,1,Eigen::Dynamic>& v,
            size_t i, size_t n) {
      validate_column_index(v, i, "segment");
      if (n != 0) validate_column_index(v, i + n - 1, "segment");
      return v.segment(i-1,n);
    }


    template <typename T>
    std::vector<T> 
    segment(const std::vector<T>& sv,
            size_t i, size_t n) {
      validate_std_vector_index(sv, i, "segment");
      if (n != 0) validate_std_vector_index(sv, i + n - 1, "segment");
      std::vector<T> s;
      for (int j = 0; j < n; ++j)
        s.push_back(sv[i + j - 1]);
      return s;
    }



  }
}

#endif
