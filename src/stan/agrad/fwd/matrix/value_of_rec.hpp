#ifndef STAN__AGRAD__FWD__MATRIX__VALUE_OF_REC_HPP
#define STAN__AGRAD__FWD__MATRIX__VALUE_OF_REC_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/functions/value_of_rec.hpp>
#include <stan/agrad/fwd/functions/value_of_rec.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace agrad {
    
    template<int R,int C>
    inline const Eigen::Matrix<double,R,C> &value_of_rec(const Eigen::Matrix<double,R,C> &M) {
      return M;
    }
    /**
     * Convert a matrix to a matrix of doubles. Matrix scalars can be 
     * arbitrarily-nested fvars and vars. fvar<var> and higher order
     * allowed through 
     * #include<stan/agrad/rev/functions/value_of_rec.hpp>
     *
     * @tparam T Scalar type in matrix
     * @tparam R Rows of matrix
     * @tparam C Columns of matrix
     * @param[in] M Matrix to be converted
     **/
    template<typename T, int R,int C>
    inline Eigen::Matrix<double,R,C> value_of_rec(const Eigen::Matrix<T,R,C> &M) {
      int i,j;
      Eigen::Matrix<double,R,C> Md(M.rows(),M.cols());
      for (j = 0; j < M.cols(); j++)
        for (i = 0; i < M.rows(); i++)
          Md(i,j) = value_of_rec(M(i,j));
      return Md;
    }
  }
}

#endif
