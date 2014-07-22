#ifndef STAN__AGRAD__REV__MATRIX__VALUE_OF_HPP
#define STAN__AGRAD__REV__MATRIX__VALUE_OF_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {
    /**
     * Convert a matrix to a matrix of doubles.  When the input is already a matrix of doubles
     * a reference is returned.
     *
     * This is used as a convenience function for implementing varis of matrix operations.
     **/
    template<int R,int C>
    inline const Eigen::Matrix<double,R,C> &value_of(const Eigen::Matrix<double,R,C> &M) {
      return M;
    }
    
    /**
     * Convert a matrix to a matrix of doubles.  When the input is already a matrix of vars
     * a new matrix is returned with the value of the constituent vars.
     *
     * This is used as a convenience function for implementing varis of matrix operations.
     **/
    template<int R,int C>
    inline Eigen::Matrix<double,R,C> value_of(const Eigen::Matrix<var,R,C> &M) {
      int i,j;
      Eigen::Matrix<double,R,C> Md(M.rows(),M.cols());
      for (j = 0; j < M.cols(); j++)
        for (i = 0; i < M.rows(); i++)
          Md(i,j) = M(i,j).vi_->val_;
      return Md;
    }
  }
}

#endif
