#ifndef STAN_MATH_PRIM_MAT_FUN_TAIL_HPP
#define STAN_MATH_PRIM_MAT_FUN_TAIL_HPP

#include <stan/math/prim/mat/err/check_column_index.hpp>
#include <stan/math/prim/mat/err/check_row_index.hpp>
#include <stan/math/prim/mat/err/check_std_vector_index.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <stan/math/prim/arr/meta/index_type.hpp>
#include <vector>

namespace stan {

  namespace math {

    /**
     * Return the specified number of elements as a vector
     * from the back of the specified vector.
     */
    template <typename T>
    inline
    Eigen::Matrix<T, Eigen::Dynamic, 1>
    tail(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v,
         size_t n) {
      if (n != 0)
        stan::math::check_row_index("tail", "n", v, n);
      return v.tail(n);
    }


    /**
     * Return the specified number of elements as a row vector
     * from the back of the specified row vector.
     */
    template <typename T>
    inline
    Eigen::Matrix<T, 1, Eigen::Dynamic>
    tail(const Eigen::Matrix<T, 1, Eigen::Dynamic>& rv,
         size_t n) {
      if (n != 0)
        stan::math::check_column_index("tail", "n", rv, n);
      return rv.tail(n);
    }

    template <typename T>
    std::vector<T> tail(const std::vector<T>& sv,
                        size_t n) {
      typedef typename index_type<std::vector<T> >::type idx_t;
      if (n != 0)
        stan::math::check_std_vector_index("tail", "n", sv, n);
      std::vector<T> s;
      for (idx_t i = sv.size() - n; i < sv.size(); ++i)
        s.push_back(sv[i]);
      return s;
    }


  }
}

#endif
