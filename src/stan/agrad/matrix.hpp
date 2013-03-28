#ifndef __STAN__AGRAD__MATRIX_HPP__
#define __STAN__AGRAD__MATRIX_HPP__


#include <stan/math/functions/Phi.hpp>
#include <stan/math/functions/logit.hpp>
#include <stan/math/matrix.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/matrix/validate_matching_sizes.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/math/matrix/validate_vector.hpp>


#include <stan/agrad/agrad.hpp>


#include <stan/agrad/rev/matrix/fill.hpp>
#include <stan/agrad/rev/matrix/Eigen_NumTraits.hpp>
#include <stan/agrad/rev/matrix/initialize_variable.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/to_var.hpp>
#include <stan/agrad/rev/matrix/dot_self.hpp>
#include <stan/agrad/rev/matrix/dot_product.hpp>
#include <stan/agrad/rev/matrix/sum.hpp>
#include <stan/agrad/rev/matrix/mdivide_left.hpp>
#include <stan/agrad/rev/matrix/mdivide_left_tri.hpp>
#include <stan/agrad/rev/matrix/determinant.hpp>
#include <stan/agrad/rev/matrix/log_determinant.hpp>
#include <stan/agrad/rev/matrix/divide.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <stan/agrad/rev/matrix/multiply_lower_tri_self_transpose.hpp>
#include <stan/agrad/rev/matrix/tcrossprod.hpp>

#include <stan/agrad/rev/matrix/assign_to_var.hpp>
#include <stan/agrad/rev/matrix/assign.hpp>

namespace stan {

  namespace agrad {
    
    /**
     * Returns the result of post-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return M times its transpose.
     */
    inline matrix_v
    tcrossprod(const matrix_v& M) {
      if (M.rows() == 0)
        return matrix_v(0,0);
      if (M.rows() == 1)
        return M * M.transpose();

      // WAS JUST THIS
      // matrix_v result(M.rows(),M.rows());
      // return result.setZero().selfadjointView<Eigen::Upper>().rankUpdate(M);

      matrix_v MMt(M.rows(),M.rows());

      vari** vs 
        = (vari**)memalloc_.alloc((M.rows() * M.cols() ) * sizeof(vari*));
      int pos = 0;
      for (int m = 0; m < M.rows(); ++m)
        for (int n = 0; n < M.cols(); ++n)
          vs[pos++] = M(m,n).vi_;
      for (int m = 0; m < M.rows(); ++m)
        MMt(m,m) = var(new dot_self_vari(vs + m * M.cols(),M.cols()));
      for (int m = 0; m < M.rows(); ++m) {
        for (int n = 0; n < m; ++n) {
          MMt(m,n) = var(new dot_product_vv_vari(vs + m * M.cols(),
                                                 vs + n * M.cols(),
                                                 M.cols()));
          MMt(n,m) = MMt(m,n);
        }
      }
      return MMt;
    }


    /**
     * Returns the result of pre-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return Transpose of M times M
     */
    inline matrix_v
    crossprod(const matrix_v& M) {
      return tcrossprod(M.transpose());
    }


    void stan_print(std::ostream* o, const var& x) {
      *o << x.val();
    }

  }
}


#endif

