#ifndef STAN__MATH__MATRIX__TAIL_HPP
#define STAN__MATH__MATRIX__TAIL_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_column_index.hpp>
#include <stan/math/error_handling/matrix/check_row_index.hpp>
#include <stan/math/error_handling/matrix/check_std_vector_index.hpp>

namespace stan {
  namespace math {

    /**
     * Return the specified number of elements as a vector
     * from the back of the specified vector.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    tail(const Eigen::Matrix<T,Eigen::Dynamic,1>& v,
         size_t n) {
      if (n != 0)
        stan::math::check_row_index("tail(%1%)",n,v,"n",(double*)0);
      return v.tail(n);
    }


    /**
     * Return the specified number of elements as a row vector
     * from the back of the specified row vector.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,1,Eigen::Dynamic>
    tail(const Eigen::Matrix<T,1,Eigen::Dynamic>& rv,
         size_t n) {
      if (n != 0)
        stan::math::check_column_index("tail(%1%)",n,rv,"n",(double*)0);
      return rv.tail(n);
    }

    template <typename T>
    std::vector<T> tail(const std::vector<T>& sv,
                        size_t n) {
      if (n != 0)
        stan::math::check_std_vector_index("tail(%1%)",n,sv,"n",(double*)0);
      std::vector<T> s;
      for (typename std::vector<T>::size_type i = sv.size() - n; i < sv.size(); ++i)
        s.push_back(sv[i]);
      return s;
    }


  }
}

#endif
