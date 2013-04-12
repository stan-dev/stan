#ifndef __STAN__AGRAD__FWD__MATRIX__MULTIPLY_HPP__
#define __STAN__AGRAD__FWD__MATRIX__MULTIPLY_HPP__

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/to_fvar.hpp>
#include <stan/agrad/fwd/matrix/dot_product.hpp>
#include <stan/agrad/fwd/operator_multiplication.hpp>

namespace stan {
  namespace agrad {
    
    template <typename T1, typename T2>
    inline
    typename stan::return_type<T1,T2>::type
    multiply(const T1& v, const T2& c) {
      return v * c;
    }

    template<typename T1, typename T2, int R1,int C1>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> 
    multiply(const Eigen::Matrix<fvar<T1>, R1, C1>& m, const fvar<T2>& c) {
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> res(m.rows(),m.cols());
      for(int i = 0; i < m.rows(); i++) {
        for(int j = 0; j < m.cols(); j++)
          res(i,j) = c * m(i,j);
      }
      return res;
    }

    template<typename T1,typename T2,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R2, C2> 
    multiply(const Eigen::Matrix<fvar<T1>, R2, C2>& m, const T2& c) {
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R2,C2> res(m.rows(),m.cols());
      for(int i = 0; i < m.rows(); i++) {
        for(int j = 0; j < m.cols(); j++)
          res(i,j) = to_fvar(c) * m(i,j);
      }
      return res;
    }

    template<typename T1, typename T2, int R1,int C1>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> 
    multiply(const Eigen::Matrix<T1, R1, C1>& m, const fvar<T2>& c) {
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> res(m.rows(),m.cols());
      for(int i = 0; i < m.rows(); i++) {
        for(int j = 0; j < m.cols(); j++)
          res(i,j) = c * to_fvar(m(i,j));
      }
      return res;
    }

    template<typename T1, typename T2, int R1,int C1>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> 
    multiply(const fvar<T1>& c, const Eigen::Matrix<fvar<T2>, R1, C1>& m) {
      return multiply(m, c);
    }

    template<typename T1, typename T2, int R1,int C1>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> 
    multiply(const T1& c, const Eigen::Matrix<fvar<T2>, R1, C1>& m) {
      return multiply(m, c);
    }

    template<typename T1, typename T2, int R1,int C1>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> 
    multiply(const fvar<T1>& c, const Eigen::Matrix<T2, R1, C1>& m) {
      return multiply(m, c);
    }
    
    template<typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2> 
    multiply(const Eigen::Matrix<fvar<T1>,R1,C1>& m1,
             const Eigen::Matrix<fvar<T2>,R2,C2>& m2) {
      stan::math::validate_multiplicable(m1,m2,"multiply");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2> result(m1.rows(),m2.cols());
      for (size_type i = 0; i < m1.rows(); i++) {
        Eigen::Matrix<fvar<T1>,1,C1> crow = m1.row(i);
        for (size_type j = 0; j < m2.cols(); j++) {
          Eigen::Matrix<fvar<T2>,R2,1> ccol = m2.col(j);
          result(i,j) = stan::agrad::dot_product(crow,ccol);
          }
        }
      return result;
    }

    template<typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2> 
    multiply(const Eigen::Matrix<fvar<T1>,R1,C1>& m1,
             const Eigen::Matrix<T2,R2,C2>& m2) {
      stan::math::validate_multiplicable(m1,m2,"multiply");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2> result(m1.rows(),m2.cols());
      for (size_type i = 0; i < m1.rows(); i++) {
        Eigen::Matrix<fvar<T1>,1,C1> crow = m1.row(i);
        for (size_type j = 0; j < m2.cols(); j++) {
          Eigen::Matrix<T2,R2,1> ccol = m2.col(j);
          result(i,j) = stan::agrad::dot_product(crow,ccol);
          }
        }
      return result;
    }

    template<typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2> 
    multiply(const Eigen::Matrix<T1,R1,C1>& m1,
             const Eigen::Matrix<fvar<T2>,R2,C2>& m2) {
      stan::math::validate_multiplicable(m1,m2,"multiply");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2> result(m1.rows(),m2.cols());
      for (size_type i = 0; i < m1.rows(); i++) {
        Eigen::Matrix<T1,1,C1> crow = m1.row(i);
        for (size_type j = 0; j < m2.cols(); j++) {
          Eigen::Matrix<fvar<T2>,R2,1> ccol = m2.col(j);
          result(i,j) = stan::agrad::dot_product(crow,ccol);
          }
        }
      return result;
    }

    template <typename T1, typename T2, int C1,int R2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type>
    multiply(const Eigen::Matrix<fvar<T1>, 1, C1>& rv, 
             const Eigen::Matrix<fvar<T2>, R2, 1>& v) {
      if (rv.size() != v.size())
        throw std::domain_error("row vector and vector must be same length in multiply");
      return dot_product(rv, v);
    }

    template <typename T1, typename T2, int C1,int R2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type>
    multiply(const Eigen::Matrix<fvar<T1>, 1, C1>& rv, 
             const Eigen::Matrix<T2, R2, 1>& v) {
      if (rv.size() != v.size())
        throw std::domain_error("row vector and vector must be same length in multiply");
      return dot_product(rv, v);
    }

    template <typename T1, typename T2, int C1,int R2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type>
    multiply(const Eigen::Matrix<T1, 1, C1>& rv, 
             const Eigen::Matrix<fvar<T2>, R2, 1>& v) {
      if (rv.size() != v.size())
        throw std::domain_error("row vector and vector must be same length in multiply");
      return dot_product(rv, v);
    }
  }
}
#endif
