#ifndef STAN__MATH__MATRIX__SPARSE_MULTIPLY_HPP
#define STAN__MATH__MATRIX__SPARSE_MULTIPLY_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/dot_product.hpp>

#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>

namespace stan {

  namespace math {

    /** Return the multiplication of the sparse matrix (specified by
     * by values and indexing) by the specified dense vector.
     *
     * The sparse matrix X of dimension m by n is represented by the
     * vector w (of values), the integer array v (containing one-based
     * column index of each value), the integer array u (containing
     * one-based indexes of where each row starts in w), and the
     * integer array z (containing the number of non-zero entries in
     * each row of w).
     *
     * @tparam T1 Type of sparse matrix entries.
     * @tparam T2 Type of dense vector entries.
     * @param m Number of rows in matrix.
     * @param n Number of columns in matrix.
     * @param w Vector of non-zero values in matrix.
     * @param v One-based column index of each non-zero value, same
     *          length as w.
     * @param u one-based index of where each row starts in w, equal to
     *          the number of rows plus one.
     * @param z number of non-zero entries in each row of w, equal to
     *          the number of rows..
     * @return dense vector for the product.
     * @throw std::domain_error if m and n are not positive or are nan.
     * @throw std::domain_error if the implied sparse matrix and b are
     *                          not multiplicable.
     * @throw std::domain_error if m/n/w/v/u/z are not internally
     * consistent, as defined by the indexing scheme.  Extractors are
     * defined in Stan which guarantee a consistent set of m/n/w/v/u/z
     * for a given sparse matrix.
     */
    template <typename T1, typename T2>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type, Eigen::Dynamic, 1>
    sparse_multiply_csr(const int& m,
                    const int& n,
                    const Eigen::Matrix<T1, Eigen::Dynamic, 1>& w,
                    const std::vector<int>& v,
                    const std::vector<int>& u,
                    const std::vector<int>& z,
                    const Eigen::Matrix<T2, Eigen::Dynamic, 1>& b) {
      using stan::math::dot_product;
      using stan::math::check_positive;
      using stan::math::check_size_match;

      check_positive("sparse_multiply_csr", "m", m);
      check_positive("sparse_multiply_csr", "n", n);
      check_size_match("sparse_multiply_csr", "n", n, "b", b.size());
      check_size_match("sparse_multiply_csr", "m", m, "u", u.size()-1);
      check_size_match("sparse_multiply_csr", "m", m, "z", z.size());
      check_size_match("sparse_multiply_csr", "w", w.size(), "v", v.size());
      check_size_match("sparse_multiply_csr", "u/z", u[m-1] + z[m-1]-1, "v", v.size());

      typedef typename boost::math::tools::promote_args<T1, T2>::type fun_scalar_t;
      Eigen::Matrix<fun_scalar_t, Eigen::Dynamic, 1>  y(m);
      for (int i = 0; i < m; ++i) {
        int end = u[i] + z[i] - 1;
        int p = 0;
        Eigen::Matrix<fun_scalar_t, Eigen::Dynamic, 1> b_sub(z[i]);
        b_sub.setZero();
        for (int q = u[i]-1; q < end; ++q) {
          b_sub(p) = b(v[q]-1);
          ++p;
        }
        Eigen::Matrix<T1, Eigen::Dynamic, 1> w_sub(w.segment(u[i]-1, z[i]));
        y(i) = dot_product(w_sub, b_sub);
      }
      return y;
    }

  }

}

#endif


