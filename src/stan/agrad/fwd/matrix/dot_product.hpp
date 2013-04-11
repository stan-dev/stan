#ifndef __STAN__AGRAD__FWD__MATRIX__DOT_PRODUCT_HPP__
#define __STAN__AGRAD__FWD__MATRIX__DOT_PRODUCT_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_vector.hpp>
#include <stan/math/matrix/validate_matching_sizes.hpp>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {

    //dot_product for vec (in matrix) * vec (in matrix); does all combos of row row, col col, row col, col row
    template<int R1,int C1, int R2, int C2>
    inline 
    fvar<double> 
    dot_product(const Eigen::Matrix<fvar<double>, R1, C1>& v1, 
                const Eigen::Matrix<fvar<double>, R2, C2>& v2) {
      stan::math::validate_vector(v1,"dot_product");
      stan::math::validate_vector(v2,"dot_product");
      stan::math::validate_matching_sizes(v1,v2,"dot_product");

      fvar<double> ret(0,0);
      for(unsigned i = 0; i < v1.size(); i++)
        ret += v1(i) * v2(i);
      return ret;
    }

    template<int R1,int C1, int R2, int C2>
    inline 
    fvar<double> 
    dot_product(const Eigen::Matrix<fvar<double>, R1, C1>& v1, 
                const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::validate_vector(v1,"dot_product");
      stan::math::validate_vector(v2,"dot_product");
      stan::math::validate_matching_sizes(v1,v2,"dot_product");

      fvar<double> ret(0,0);
      for(unsigned i = 0; i < v1.size(); i++)
        ret += v1(i) * v2(i);
      return ret;
    }

    template<int R1,int C1, int R2, int C2>
    inline 
    fvar<double> 
    dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                const Eigen::Matrix<fvar<double>, R2, C2>& v2) {
      stan::math::validate_vector(v1,"dot_product");
      stan::math::validate_vector(v2,"dot_product");
      stan::math::validate_matching_sizes(v1,v2,"dot_product");

      fvar<double> ret(0,0);
      for(unsigned i = 0; i < v1.size(); i++)
        ret += v1(i) * v2(i);
      return ret;
    }



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



    inline 
    fvar<double>
    dot_product(const std::vector<fvar<double> >& v1,
                const std::vector<fvar<double> >& v2) {
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      fvar<double> ret(0,0);
      for(unsigned i = 0; i < v1.size(); i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }

    inline 
    fvar<double>
    dot_product(const std::vector<double>& v1,
                const std::vector<fvar<double> >& v2) {
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      fvar<double> ret(0,0);
      for(unsigned i = 0; i < v1.size(); i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }

    inline 
    fvar<double>
    dot_product(const std::vector<fvar<double> >& v1,
                const std::vector<double>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      fvar<double> ret(0,0);
      for(unsigned i = 0; i < v1.size(); i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }

    template<int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<double>, 1, C1>
    columns_dot_product(const Eigen::Matrix<fvar<double>, R1, C1>& v1, 
                        const Eigen::Matrix<fvar<double>, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<fvar<double>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        ret(j) = dot_product(v1.col(j),v2.col(j));
      }
      return ret;
    }
    
    template<int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<double>, 1, C1>
    columns_dot_product(const Eigen::Matrix<fvar<double>, R1, C1>& v1, 
                        const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<fvar<double>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        ret(j) = dot_product(v1.col(j),v2.col(j));
      }
      return ret;
    }

    template<int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<double>, 1, C1>
    columns_dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                        const Eigen::Matrix<fvar<double>, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<fvar<double>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        ret(j) = dot_product(v1.col(j),v2.col(j));
      }
      return ret;
    }

    template<int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<double>, R1, 1>
    rows_dot_product(const Eigen::Matrix<fvar<double>, R1, C1>& v1, 
                     const Eigen::Matrix<fvar<double>, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"rows_dot_product");
      Eigen::Matrix<fvar<double>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        ret(j) = dot_product(v1.rows(j),v2.rows(j));
      }
      return ret;
    }
    
    template<int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<double>, R1, 1>
    rows_dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                     const Eigen::Matrix<fvar<double>, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"rows_dot_product");
      Eigen::Matrix<fvar<double>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        ret(j) = dot_product(v1.rows(j),v2.rows(j));
      }
      return ret;
    }

    template<int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<double>, R1, 1>
    rows_dot_product(const Eigen::Matrix<fvar<double>, R1, C1>& v1, 
                     const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"rows_dot_product");
      Eigen::Matrix<fvar<double>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        ret(j) = dot_product(v1.rows(j),v2.rows(j));
      }
      return ret;
    }
  }
}
#endif
