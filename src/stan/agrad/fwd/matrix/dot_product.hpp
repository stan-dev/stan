#ifndef __STAN__AGRAD__FWD__MATRIX__DOT_PRODUCT_HPP__
#define __STAN__AGRAD__FWD__MATRIX__DOT_PRODUCT_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_vector.hpp>
#include <stan/math/matrix/validate_matching_sizes.hpp>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/to_fvar.hpp>

namespace stan {
  namespace agrad {

    //dot_product for vec (in matrix) * vec (in matrix); does all combos of row row, col col, row col, col row
    template<typename T1, typename T2, int R1,int C1, int R2, int C2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type> 
    dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_vector(v1,"dot_product");
      stan::math::validate_vector(v2,"dot_product");
      stan::math::validate_matching_sizes(v1,v2,"dot_product");

      fvar<typename stan::return_type<T1,T2>::type> ret(0,0);
      for(size_type i = 0; i < v1.size(); i++)
        ret += v1(i) * v2(i);
      return ret;
    }

    template<typename T1, typename T2, int R1,int C1, int R2, int C2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type> 
    dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                const Eigen::Matrix<T2, R2, C2>& v2) {
      stan::math::validate_vector(v1,"dot_product");
      stan::math::validate_vector(v2,"dot_product");
      stan::math::validate_matching_sizes(v1,v2,"dot_product");

      fvar<typename stan::return_type<T1,T2>::type> ret(0,0);
      for(size_type i = 0; i < v1.size(); i++)
        ret += v1(i) * to_fvar(v2(i));
      return ret;
    }

    template<typename T1, typename T2, int R1,int C1, int R2, int C2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type> 
    dot_product(const Eigen::Matrix<T1, R1, C1>& v1, 
                const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_vector(v1,"dot_product");
      stan::math::validate_vector(v2,"dot_product");
      stan::math::validate_matching_sizes(v1,v2,"dot_product");

      fvar<typename stan::return_type<T1,T2>::type> ret(0,0);
      for(size_type i = 0; i < v1.size(); i++)
        ret += to_fvar(v1(i)) * v2(i);
      return ret;
    }


    //not sure what this is for..
    // /**
    //  * Returns the dot product.
    //  *
    //  * @param[in] v1 First array.
    //  * @param[in] v2 Second array.
    //  * @param[in] length Length of both arrays.
    //  * @return Dot product of the arrays.
    //  */
    // inline var dot_product(const var* v1, const var* v2, size_t length) {
    //   return var(new dot_product_vv_vari(v1, v2, length));
    // }
    // /**
    //  * Returns the dot product.
    //  *
    //  * @param[in] v1 First array.
    //  * @param[in] v2 Second array.
    //  * @param[in] length Length of both arrays.
    //  * @return Dot product of the arrays.
    //  */
    // inline var dot_product(const var* v1, const double* v2, size_t length) {
    //   return var(new dot_product_vd_vari(v1, v2, length));
    // }
    // /**
    //  * Returns the dot product.
    //  *
    //  * @param[in] v1 First array.
    //  * @param[in] v2 Second array.
    //  * @param[in] length Length of both arrays.
    //  * @return Dot product of the arrays.
    //  */
    // inline var dot_product(const double* v1, const var* v2, size_t length) {
    //   return var(new dot_product_vd_vari(v2, v1, length));
    // }

    template<typename T1, typename T2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type>
    dot_product(const std::vector<fvar<T1> >& v1,
                const std::vector<fvar<T2> >& v2) {
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      fvar<typename stan::return_type<T1,T2>::type> ret(0,0);
      for(size_type i = 0; i < v1.size(); i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }

    template<typename T1, typename T2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type>
    dot_product(const std::vector<T1>& v1,
                const std::vector<fvar<T2> >& v2) {
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      fvar<typename stan::return_type<T1,T2>::type> ret(0,0);
      for(size_type i = 0; i < v1.size(); i++)
        ret += to_fvar(v1.at(i)) * v2.at(i);
      return ret;
    }

    template<typename T1, typename T2>
    inline 
    fvar<typename stan::return_type<T1, T2>::type>
    dot_product(const std::vector<fvar<T1> >& v1,
                const std::vector<T2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      fvar<typename stan::return_type<T1,T2>::type> ret(0,0);
      for(size_type i = 0; i < v1.size(); i++)
        ret += v1.at(i) * to_fvar(v2.at(i));
      return ret;
    }

    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1>
    columns_dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                        const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        Eigen::Matrix<fvar<T1>,R1,1> ccol1 = v1.col(j);
        Eigen::Matrix<fvar<T2>,R2,1> ccol2 = v2.col(j);
        ret(0,j) = dot_product(ccol1, ccol2);
      }
      return ret;
    }
    
    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1>
    columns_dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                        const Eigen::Matrix<T2, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        Eigen::Matrix<fvar<T1>,R1,1> ccol1 = v1.col(j);
        Eigen::Matrix<T2,R2,1> ccol = v2.col(j);
        Eigen::Matrix<fvar<T2>,R2,1> ccol2 = to_fvar(ccol);
        ret(0,j) = dot_product(ccol1, ccol2);
      }
      return ret;
    }

    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1>
    columns_dot_product(const Eigen::Matrix<T1, R1, C1>& v1, 
                        const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        Eigen::Matrix<T1,R1,1> ccol = v1.col(j);
        Eigen::Matrix<fvar<T1>,R1,1> ccol1 = to_fvar(ccol);
        Eigen::Matrix<fvar<T2>,R2,1> ccol2 = v2.col(j);
        ret(0,j) = dot_product(ccol1, ccol2);
      }
      return ret;
    }

    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1>
    rows_dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                     const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"rows_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        Eigen::Matrix<fvar<T1>,1,C1> crow1 = v1.row(j);
        Eigen::Matrix<fvar<T2>,1,C2> crow2 = v2.row(j);
        ret(j,0) = dot_product(crow1, crow2);
      }
      return ret;
    }
    
    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1>
    rows_dot_product(const Eigen::Matrix<T1, R1, C1>& v1, 
                     const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"rows_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        Eigen::Matrix<T1,1,C1> crow = v1.row(j);
        Eigen::Matrix<fvar<T1>,1,C1> crow1 = to_fvar(crow);
        Eigen::Matrix<fvar<T2>,1,C2> crow2 = v2.row(j);
        ret(j,0) = dot_product(crow1, crow2);
      }
      return ret;
    }

    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1>
    rows_dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                     const Eigen::Matrix<T2, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"rows_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        Eigen::Matrix<fvar<T1>,1,C1> crow1 = v1.row(j);
        Eigen::Matrix<T2,1,C2> crow = v2.row(j);
        Eigen::Matrix<fvar<T2>,1,C2> crow2 = to_fvar(crow);
        ret(j,0) = dot_product(crow1, crow2);
      }
      return ret;
    }
  }
}
#endif
