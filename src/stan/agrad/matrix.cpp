#ifndef __STAN__AGRAD__MATRIX_CPP__
#define __STAN__AGRAD__MATRIX_CPP__

#include "stan/agrad/matrix.hpp"

namespace stan {

  namespace agrad {

    var determinant(const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& m) {
      if (m.rows() != m.cols())
        throw std::domain_error ("m must be a square matrix");
      return m.determinant();
    }

    row_vector_v row(const matrix_v& m, size_t i) {
      if (i == 0U) {
        throw std::domain_error("row() indexes from 1; found index i=0");
      }
      if (i > static_cast<size_t>(m.rows())) {
        std::stringstream msg;
        msg << "index must be less than or equal to number of rows"
            << " found m.rows()=" << m.rows()
            << "; i=" << i;
        throw std::domain_error(msg.str());
      }
      return m.row(i - 1);
    }

    vector_v col(const matrix_v& m, size_t j) {
      if (j == 0U) {
        throw std::domain_error("row() indexes from 1; found index i=0");
      }
      if (j > static_cast<size_t>(m.cols())) {
        std::stringstream msg;
        msg << "index must be less than or equal to number of rows"
            << " found m.cols()=" << m.cols()
            << "; =" << j;
        throw std::domain_error(msg.str());
      }
      return m.col(j - 1);
    }

    vector_v diagonal(const matrix_v& m) {
      return m.diagonal();
    }

    matrix_v diag_matrix(const vector_v& v) {
      return v.asDiagonal();
    }
    
    row_vector_v transpose(const vector_v& v) {
      return v.transpose();
    }

    vector_v transpose(const row_vector_v& rv) {
      return rv.transpose();
    }

    matrix_v transpose(const matrix_v& m) {
      return m.transpose();
    }

    matrix_v inverse(const matrix_v& m) {
      return m.inverse();
    }

    vector_v softmax(const vector_v& x) {
      vector_v theta(x.size());
      stan::math::softmax<vector_v,stan::agrad::var>(x,theta);
      return theta;
    }

    vector_v eigenvalues(const matrix_v& m) {
      // false == no vectors
      Eigen::EigenSolver<matrix_v> solver(m,false);
      // FIXME: test imag() all 0?
      return solver.eigenvalues().real();
    }

    matrix_v eigenvectors(const matrix_v& m) {
      Eigen::EigenSolver<matrix_v> solver(m);
      return solver.eigenvectors().real();
    }

    void eigen_decompose(const matrix_v& m,
                         vector_v& eigenvalues,
                         matrix_v& eigenvectors) {
      Eigen::EigenSolver<matrix_v> solver(m);
      eigenvalues = solver.eigenvalues().real();
      eigenvectors = solver.eigenvectors().real();
    }

    vector_v eigenvalues_sym(const matrix_v& m) {
      Eigen::SelfAdjointEigenSolver<matrix_v> solver(m,Eigen::EigenvaluesOnly);
      return solver.eigenvalues().real();
    }

    matrix_v eigenvectors_sym(const matrix_v& m) {
      Eigen::SelfAdjointEigenSolver<matrix_v> solver(m);
      return solver.eigenvectors().real();
    }

    void eigen_decompose_sym(const matrix_v& m,
                             vector_v& eigenvalues,
                             matrix_v& eigenvectors) {
      Eigen::SelfAdjointEigenSolver<matrix_v> solver(m);
      eigenvalues = solver.eigenvalues().real();
      eigenvectors = solver.eigenvectors().real();
    }

    matrix_v cholesky_decompose(const matrix_v& m) {
      if (m.rows() != m.cols()) {
        throw std::domain_error ("m must be a square matrix");
      }
      Eigen::LLT<matrix_v> llt(m.rows());
      llt.compute(m);
      return llt.matrixL();
    }

    vector_v singular_values(const matrix_v& m) {
      Eigen::JacobiSVD<matrix_v> svd(m); // no U or V
      return svd.singularValues();
    }      

    void svd(const matrix_v& m,
             matrix_v& u,
             matrix_v& v,
             vector_v& s) {
      static const unsigned int THIN_SVD_OPTIONS
        = Eigen::ComputeThinU | Eigen::ComputeThinV;
      Eigen::JacobiSVD<matrix_v> svd(m, THIN_SVD_OPTIONS);
      u = svd.matrixU();
      v = svd.matrixV();
      s = svd.singularValues();
    }

    void stan_print(std::ostream* o, const var& x) {
      *o << x.val();
    }


  }

}

#endif
