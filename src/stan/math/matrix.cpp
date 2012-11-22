#ifndef __STAN__MATH__MATRIX_CPP__
#define __STAN__MATH__MATRIX_CPP__

#include "stan/math/matrix.hpp"
#include "stan/math/special_functions.hpp"

namespace stan {

  namespace math {



    matrix_d inverse(const matrix_d& m) {
      validate_square(m,"matirx inverse");
      return m.inverse();
    }

    vector_d
    softmax(const vector_d& x) {
      validate_nonzero_size(x,"vector softmax");
      vector_d theta(x.size());
      softmax<vector_d,double>(x,theta);
      return theta;
    }

    vector_d eigenvalues(const matrix_d& m) {
      validate_square(m,"eigenvalues");
      // false arg means no eigenvector calcs
      Eigen::EigenSolver<matrix_d> solver(m,false);
      // FIXME: test imag() all 0?
      return solver.eigenvalues().real();
    }

    matrix_d eigenvectors(const matrix_d& m) {
      validate_nonzero_size(m,"eigenvectors");
      validate_square(m,"eigenvectors");
      Eigen::EigenSolver<matrix_d> solver(m);
      return solver.eigenvectors().real();
    }

    void eigen_decompose(const matrix_d& m,
                         vector_d& eigenvalues,
                         matrix_d& eigenvectors) {
      Eigen::EigenSolver<matrix_d> solver(m);
      eigenvalues = solver.eigenvalues().real();
      eigenvectors = solver.eigenvectors().real();
    }

    vector_d eigenvalues_sym(const matrix_d& m) {
      validate_nonzero_size(m,"eigenvalues_sym");
      validate_square(m,"eigenvalues_sym");
      // FIXME: validate actually symmetric?
      Eigen::SelfAdjointEigenSolver<matrix_d> solver(m,Eigen::EigenvaluesOnly);
      return solver.eigenvalues().real();
    }

    matrix_d eigenvectors_sym(const matrix_d& m) {
      validate_nonzero_size(m,"eigenvectors_sym");
      validate_square(m,"eigenvectors_sym");
      Eigen::SelfAdjointEigenSolver<matrix_d> solver(m);
      return solver.eigenvectors().real();
    }
  
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
