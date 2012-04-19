#ifndef __STAN__MATH__MATRIX_CPP__
#define __STAN__MATH__MATRIX_CPP__

#include "stan/math/matrix.hpp"

namespace stan {

  namespace math {

    double determinant(const matrix_d &m) {
      return m.determinant();
    }

    vector_d add(const vector_d& v1, const vector_d& v2) {
      if (v1.size() != v2.size()) 
        throw std::invalid_argument ("v1.size() != v2.size()");
      return v1 + v2;
    }

    row_vector_d add(const row_vector_d& rv1, const row_vector_d& rv2) {
      if (rv1.size() != rv2.size()) 
        throw std::invalid_argument ("rv1.size() != rv2.size()");
      return rv1 + rv2;
    }

    matrix_d add(const matrix_d& m1, const matrix_d& m2) {
      if (m1.rows() != m2.rows() || m1.cols() != m2.cols())
        throw std::invalid_argument ("dimensions of m1 and m2 do not match");
      return m1 + m2;
    }

    vector_d subtract(const vector_d& v1, const vector_d& v2) {
      if (v1.size() != v2.size())
        throw std::invalid_argument ("v1.size() != v2.size()");
      return v1 - v2;
    }

    row_vector_d subtract(const row_vector_d& rv1, const row_vector_d& rv2) {
      if (rv1.size() != rv2.size())
        throw std::invalid_argument ("rv1.size() != rv2.size()");
      return rv1 - rv2;
    }

    matrix_d subtract(const matrix_d& m1, const matrix_d& m2) {
      if (m1.rows() != m2.rows() || m1.cols() != m2.cols())
        throw std::invalid_argument ("dimensions of m1 and m2 do not match");
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
      if (v1.size() != v2.size()) {
        std::stringstream msg;
        msg << "vectors must have same dimensions, v1.size()=" << v1.size()
            << ", v2.size()=" << v2.size() << std::endl;
        throw std::domain_error(msg.str());
      }
      vector_d prod(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        prod(i) = v1(i) * v2(i);
      return prod;
    }

    row_vector_d elt_multiply(const row_vector_d& v1, const row_vector_d& v2) {
      if (v1.size() != v2.size()) {
        std::stringstream msg;
        msg << "vectors must have same dimensions, v1.size()=" << v1.size()
            << ", v2.size()=" << v2.size() << std::endl;
        throw std::domain_error(msg.str());
      }
      row_vector_d prod(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        prod(i) = v1(i) * v2(i);
      return prod;
    }

    matrix_d elt_multiply(const matrix_d& m1, const matrix_d& m2) {
      if (m1.rows() != m2.rows()
          || m1.cols() != m2.cols()) {
        std::stringstream msg;
        msg << "vectors must have same dimensions"
            << "; m1.rows()=" << m1.rows() 
            << ", m2.rows()=" << m2.rows()
            << ", m1.cols()=" << m1.cols()
            << ", m2.cols()=" << m2.cols()
            << std::endl;
        throw std::domain_error(msg.str());
      }
      matrix_d prod(m1.rows(),m1.cols());
      for (int j = 0; j < m1.cols(); ++j)
        for (int i = 0; i < m1.rows(); ++i)
          prod(i,j) = m1(i,j) * m2(i,j);
      return prod;
    }

    vector_d elt_divide(const vector_d& v1, const vector_d& v2) {
      if (v1.size() != v2.size()) {
        std::stringstream msg;
        msg << "require vectors to be same size for element-wise division;"
            << " found v1.size()=" << v1.size()
            << ", v2.size()=" << v2.size()
            << std::endl;
        throw std::domain_error(msg.str());
      }
      vector_d prod(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        prod(i) = v1(i) / v2(i);
      return prod;
    }

    row_vector_d elt_divide(const row_vector_d& v1, const row_vector_d& v2) {
      if (v1.size() != v2.size()) {
        std::stringstream msg;
        msg << "require vectors to be same size for element-wise division;"
            << " found v1.size()=" << v1.size()
            << ", v2.size()=" << v2.size()
            << std::endl;
        throw std::domain_error(msg.str());
      }
      row_vector_d prod(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        prod(i) = v1(i) / v2(i);
      return prod;
    }

    matrix_d elt_divide(const matrix_d& m1, const matrix_d& m2) {
      if (m1.rows() != m2.rows() 
          || m1.cols() != m2.cols()) {
        std::stringstream msg;
        msg << "require matrices to be same dimensions for element-wise division;"
            << " found m1.rows()=" << m1.rows() << ", m1.cols()=" << m1.cols()
            << "; m2.rows()=" << m2.rows() << ", m2.cols()=" << m2.cols()
            << std::endl;
        throw std::domain_error(msg.str());

      }
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
      return m.row(i - 1);
    }

    vector_d col(const matrix_d& m, size_t j) {
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
      return m.inverse();
    }

    vector_d eigenvalues(const matrix_d& m) {
      // false == no vectors
      Eigen::EigenSolver<matrix_d> solver(m,false);
      // FIXME: test imag() all 0?
      return solver.eigenvalues().real();
    }

    matrix_d eigenvectors(const matrix_d& m) {
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
      Eigen::SelfAdjointEigenSolver<matrix_d> solver(m,Eigen::EigenvaluesOnly);
      return solver.eigenvalues().real();
    }

    matrix_d eigenvectors_sym(const matrix_d& m) {
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
      if (m.rows() != m.cols())
        throw std::invalid_argument ("m must be a square matrix");
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
