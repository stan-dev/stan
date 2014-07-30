#ifndef STAN__AGRAD__REV__MATRIX__TCROSSPROD_HPP
#define STAN__AGRAD__REV__MATRIX__TCROSSPROD_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/Eigen_NumTraits.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/dot_product.hpp>
#include <stan/agrad/rev/matrix/dot_self.hpp>

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
      // if (M.rows() == 1)
      //   return M * M.transpose();

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
          MMt(m,n) = var(new dot_product_vari<var,var>(vs + m * M.cols(),
                                                       vs + n * M.cols(),
                                                       M.cols()));
          MMt(n,m) = MMt(m,n);
        }
      }
      return MMt;
    }

  }
}
#endif
