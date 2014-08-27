#ifndef STAN__MATH__MATRIX__EXP_HPP
#define STAN__MATH__MATRIX__EXP_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace math {
    
    /**
     * Return the element-wise exponentiation of the matrix or vector.
     *
     * @param m The matrix or vector.
     * @return ret(i,j) = exp(m(i,j))
     */
    template<typename T, int Rows, int Cols>
    inline Eigen::Matrix<T,Rows,Cols> exp(const Eigen::Matrix<T,Rows,Cols> & m) {
      return m.array().exp().matrix();
    }
    
    template<int Rows, int Cols>
    inline Eigen::Matrix<double,Rows,Cols> exp(const Eigen::Matrix<double,Rows,Cols> & m) {
      for (const double * it = m.data(), * end_ = it + m.size(); it != end_; it++)
        if (boost::math::isnan(*it)) {
          Eigen::Matrix<double,Rows,Cols> mat = m;
          for (double * it = mat.data(), * end_ = it + mat.size(); it != end_; it++)
            *it = std::exp(*it);
          return mat;
        }
      return m.array().exp().matrix();
    }
    
  }
}
#endif
