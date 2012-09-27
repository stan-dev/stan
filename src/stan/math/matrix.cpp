#ifndef __STAN__MATH__MATRIX_CPP__
#define __STAN__MATH__MATRIX_CPP__

#include "stan/math/matrix.hpp"
#include "stan/math/special_functions.hpp"

namespace stan {

  namespace math {

    double determinant(const matrix_d &m) {
      validate_square(m,"determinant");
      return m.determinant();
    }

    vector_d add(const vector_d& v1, const vector_d& v2) {
      validate_matching_dims(v1,v2,"add");
      return v1 + v2;
    }

    row_vector_d add(const row_vector_d& rv1, const row_vector_d& rv2) {
      validate_matching_dims(rv1,rv2,"add");
      return rv1 + rv2;
    }

    matrix_d add(const matrix_d& m1, const matrix_d& m2) {
      validate_matching_dims(m1,m2,"add");
      return m1 + m2;
    }

    vector_d subtract(const vector_d& v1, const vector_d& v2) {
      validate_matching_dims(v1,v2,"subtract");
      return v1 - v2;
    }

    row_vector_d subtract(const row_vector_d& rv1, const row_vector_d& rv2) {
      validate_matching_dims(rv1,rv2,"subtract");
      return rv1 - rv2;
    }

    matrix_d subtract(const matrix_d& m1, const matrix_d& m2) {
      validate_matching_dims(m1,m2,"subtract");
      return m1 - m2;
    }

    vector_d minus(const vector_d& v) {
      return -v;
    }
  
    row_vector_d minus(const row_vector_d& rv) {
      return -rv;
    }

    matrix_d minus(const matrix_d& m) {
      return -m;
    }

    vector_d divide(const vector_d& v, double c) {
      return v / c;
    }

    row_vector_d divide(const row_vector_d& rv, double c) {
      return rv / c;
    }

    matrix_d divide(const matrix_d& m, double c) {
      return m / c;
    }

    vector_d elt_multiply(const vector_d& v1, const vector_d& v2) {
      validate_matching_dims(v1,v2,"elementwise multiply");
      vector_d prod(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        prod(i) = v1(i) * v2(i);
      return prod;
    }

    row_vector_d elt_multiply(const row_vector_d& v1, const row_vector_d& v2) {
      validate_matching_dims(v1,v2,"elementwise multiply");
      row_vector_d prod(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        prod(i) = v1(i) * v2(i);
      return prod;
    }

    matrix_d elt_multiply(const matrix_d& m1, const matrix_d& m2) {
      validate_matching_dims(m1,m2,"elementwise multiply");
      matrix_d prod(m1.rows(),m1.cols());
      for (int j = 0; j < m1.cols(); ++j)
        for (int i = 0; i < m1.rows(); ++i)
          prod(i,j) = m1(i,j) * m2(i,j);
      return prod;
    }

    vector_d elt_divide(const vector_d& v1, const vector_d& v2) {
      validate_matching_dims(v1,v2,"elementwise divide");
      vector_d prod(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        prod(i) = v1(i) / v2(i);
      return prod;
    }

    row_vector_d elt_divide(const row_vector_d& v1, const row_vector_d& v2) {
      validate_matching_dims(v1,v2,"elementwise divide");
      row_vector_d prod(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        prod(i) = v1(i) / v2(i);
      return prod;
    }

    matrix_d elt_divide(const matrix_d& m1, const matrix_d& m2) {
      validate_matching_dims(m1,m2,"elementwise divide");
      matrix_d prod(m1.rows(),m1.cols());
      for (int j = 0; j < m2.cols(); ++j)
        for (int i = 0; i < m1.rows(); ++i)
          prod(i,j) = m1(i,j) / m2(i,j);
      return prod;
    }

    vector_d multiply(const vector_d& v, double c) {
      return c * v;
    }

    row_vector_d multiply(const row_vector_d& rv, double c) {
      return c * rv;
    }

    matrix_d multiply(const matrix_d& m, double c) {
      return c * m;
    }

    vector_d multiply(double c, const vector_d& v) {
      return c * v;
    }

    row_vector_d multiply(double c, const row_vector_d& rv) {
      return c * rv;
    }

    matrix_d multiply(double c, const matrix_d& m) {
      return c * m;
    }

    row_vector_d row(const matrix_d& m, size_t i) {
      validate_row_index(m,i,"row");
      return m.row(i - 1);
    }

    vector_d col(const matrix_d& m, size_t j) {
      validate_column_index(m,j,"col");
      return m.col(j - 1);
    }

    vector_d diagonal(const matrix_d& m) {
      return m.diagonal();
    }

    matrix_d diag_matrix(const vector_d& v) {
      return v.asDiagonal();
    }

    row_vector_d transpose(const vector_d& v) {
      return v.transpose();
    }

    vector_d transpose(const row_vector_d& rv) {
      return rv.transpose();
    }

    matrix_d transpose(const matrix_d& m) {
      return m.transpose();
    }

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
