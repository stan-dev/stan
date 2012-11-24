#ifndef __STAN__MATH__MATRIX_CPP__
#define __STAN__MATH__MATRIX_CPP__

#include "stan/math/matrix.hpp"
#include "stan/math/special_functions.hpp"

namespace stan {

  namespace math {



    void eigen_decompose_sym(const matrix_d& m,
                             vector_d& eigenvalues,
                             matrix_d& eigenvectors) {
      Eigen::SelfAdjointEigenSolver<matrix_d> solver(m);
      eigenvalues = solver.eigenvalues().real();
      eigenvectors = solver.eigenvectors().real();
    }

    matrix_d cholesky_decompose(const matrix_d& m) {
      validate_square(m,"cholesky decomposition");
      Eigen::LLT<matrix_d> llt(m.rows());
      llt.compute(m);
      return llt.matrixL();
    }
  
    vector_d singular_values(const matrix_d& m) {
      Eigen::JacobiSVD<matrix_d> svd(m); // no U or V
      return svd.singularValues();
    }      

    void svd(const matrix_d& m, matrix_d& u, matrix_d& v, vector_d& s) {
      static const unsigned int THIN_SVD_OPTIONS
        = Eigen::ComputeThinU | Eigen::ComputeThinV;
      Eigen::JacobiSVD<matrix_d> svd(m, THIN_SVD_OPTIONS);
      u = svd.matrixU();
      v = svd.matrixV();
      s = svd.singularValues();
    }

  }

}

#endif
