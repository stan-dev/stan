#ifndef STAN_MATH_PRIM_MAT_FUN_ASSIGN_HPP
#define STAN_MATH_PRIM_MAT_FUN_ASSIGN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/err/invalid_argument.hpp>
#include <stan/math/prim/mat/err/check_matching_sizes.hpp>
#include <stan/math/prim/mat/err/check_matching_dims.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {

  namespace math {

    // Recursive assignment with size match checking and promotion

    /**
     * Copy the right-hand side's value to the left-hand side
     * variable.
     *
     * The <code>assign()</code> function is overloaded.  This
     * instance will match arguments where the right-hand side is
     * assignable to the left and they are not both
     * <code>std::vector</code> or <code>Eigen::Matrix</code> types.
     *
     * @tparam LHS Type of left-hand side.
     * @tparam RHS Type of right-hand side.
     * @param lhs Left-hand side.
     * @param rhs Right-hand side.
     */
    template <typename LHS, typename RHS>
    inline void
    assign(LHS& lhs, const RHS& rhs) {
      lhs = rhs;
    }

    /**
     * Copy the right-hand side's value to the left-hand side
     * variable.
     *
     * The <code>assign()</code> function is overloaded.  This
     * instance will be called for arguments that are both
     * <code>Eigen::Matrix</code> types, but whose shapes  are not
     * compatible.  The shapes are specified in the row and column
     * template parameters.
     *
     * @tparam LHS Type of left-hand side matrix elements.
     * @tparam RHS Type of right-hand side matrix elements.
     * @tparam R1 Row shape of left-hand side matrix.
     * @tparam C1 Column shape of left-hand side matrix.
     * @tparam R2 Row shape of right-hand side matrix.
     * @tparam C2 Column shape of right-hand side matrix.
     * @param x Left-hand side matrix.
     * @param y Right-hand side matrix.
     * @throw std::invalid_argument
     */
    template <typename LHS, typename RHS, int R1, int C1, int R2, int C2>
    inline void
    assign(Eigen::Matrix<LHS, R1, C1>& x,
           const Eigen::Matrix<RHS, R2, C2>& y) {
      std::stringstream ss;
      ss << "shapes must match, but found"
         << " R1=" << R1
         << "; C1=" << C1
         << "; R2=" << R2
         << "; C2=" << C2;
      std::string ss_str(ss.str());
      invalid_argument("assign(Eigen::Matrix, Eigen::Matrix)",
                       "", "", ss_str.c_str());
    }

    /**
     * Copy the right-hand side's value to the left-hand side
     * variable.
     *
     * The <code>assign()</code> function is overloaded.  This
     * instance will be called for arguments that are both
     * <code>Eigen::Matrix</code> types and whose shapes match.  The
     * shapes are specified in the row and column template parameters.
     *
     * @tparam LHS Type of left-hand side matrix elements.
     * @tparam RHS Type of right-hand side matrix elements.
     * @tparam R Row shape of both matrices.
     * @tparam C Column shape of both mtarices.
     * @param x Left-hand side matrix.
     * @param y Right-hand side matrix.
     * @throw std::invalid_argument if sizes do not match.
     */
    template <typename LHS, typename RHS, int R, int C>
    inline void
    assign(Eigen::Matrix<LHS, R, C>& x,
           const Eigen::Matrix<RHS, R, C>& y) {
      stan::math::check_matching_dims("assign",
                                                "x", x,
                                                "y", y);
      for (int i = 0; i < x.size(); ++i)
        assign(x(i), y(i));
    }

    /**
     * Copy the right-hand side's value to the left-hand side
     * variable.
     *
     * The <code>assign()</code> function is overloaded.  This
     * instance will be called for arguments that are both
     * <code>Eigen::Matrix</code> types and whose shapes match.  The
     * shape of the right-hand side matrix is specified in the row and
     * column shape template parameters.
     *
     * @tparam LHS Type of matrix block elements.
     * @tparam RHS Type of right-hand side matrix elements.
     * @tparam R Row shape for right-hand side matrix.
     * @tparam C Column shape for right-hand side matrix.
     * @param x Left-hand side block view of matrix.
     * @param y Right-hand side matrix.
     * @throw std::invalid_argument if sizes do not match.
     */
    template <typename LHS, typename RHS, int R, int C>
    inline void
    assign(Eigen::Block<LHS> x,
           const Eigen::Matrix<RHS, R, C>& y) {
      stan::math::check_matching_sizes("assign",
                                                 "x", x,
                                                 "y", y);
      for (int n = 0; n < y.cols(); ++n)
        for (int m = 0; m < y.rows(); ++m)
          assign(x(m, n), y(m, n));
    }


    /**
     * Copy the right-hand side's value to the left-hand side
     * variable.
     *
     * The <code>assign()</code> function is overloaded.  This
     * instance will be called for arguments that are both
     * <code>std::vector</code>, and will call <code>assign()</code>
     * element-by element.
     *
     * For example, a <code>std::vector&lt;int&gt;</code> can be
     * assigned to a <code>std::vector&lt;double&gt;</code> using this
     * function.
     *
     * @tparam LHS Type of left-hand side vector elements.
     * @tparam RHS Type of right-hand side vector elements.
     * @param x Left-hand side vector.
     * @param y Right-hand side vector.
     * @throw std::invalid_argument if sizes do not match.
     */
    template <typename LHS, typename RHS>
    inline void
    assign(std::vector<LHS>& x, const std::vector<RHS>& y) {
      stan::math::check_matching_sizes("assign",
                                                 "x", x,
                                                 "y", y);
      for (size_t i = 0; i < x.size(); ++i)
        assign(x[i], y[i]);
    }


  }
}
#endif
