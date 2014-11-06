#ifndef STAN__AGRAD__REV__MATRIX__MULTIPLY_LOWER_TRI_SELF_TRANSPOSE_HPP
#define STAN__AGRAD__REV__MATRIX__MULTIPLY_LOWER_TRI_SELF_TRANSPOSE_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/dot_product.hpp>
#include <stan/agrad/rev/matrix/dot_self.hpp>

namespace stan {
  namespace agrad {
    
    inline matrix_v 
    multiply_lower_tri_self_transpose(const matrix_v& L) {
      //stan::error_handling::check_square("multiply_lower_tri_self_transpose",
      //L,"L",(double*)0);
      int K = L.rows();
      int J = L.cols();
      matrix_v LLt(K,K);
      if (K == 0) return LLt;
      // if (K == 1) {
      //   LLt(0,0) = L(0,0) * L(0,0);
      //   return LLt;
      // }
      int Knz;
      if (K >= J)
        Knz = (K-J)*J + (J * (J + 1)) / 2;
      else // if (K < J)
        Knz = (K * (K + 1)) / 2;
      vari** vs = (vari**)memalloc_.alloc( Knz * sizeof(vari*) );
      int pos = 0;
      for (int m = 0; m < K; ++m)
        for (int n = 0; n < ((J < (m+1))?J:(m+1)); ++n) {
          vs[pos++] = L(m,n).vi_;
        }
      for (int m = 0, mpos=0; m < K; ++m, mpos += (J < m)?J:m) {
        LLt(m,m) = var(new dot_self_vari(vs + mpos, (J < (m+1))?J:(m+1)));
        for (int n = 0, npos = 0; n < m; ++n, npos += (J < n)?J:n) {
          LLt(m,n) = LLt(n,m) = var(new dot_product_vari<var,var>(vs + mpos, vs + npos, (J < (n+1))?J:(n+1)));
        }
      }
      return LLt;
    }

  }
}
#endif
