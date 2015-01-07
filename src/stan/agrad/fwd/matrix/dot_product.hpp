#ifndef STAN__AGRAD__FWD__MATRIX__DOT_PRODUCT_HPP
#define STAN__AGRAD__FWD__MATRIX__DOT_PRODUCT_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/error_handling/matrix/check_vector.hpp>
#include <stan/error_handling/matrix/check_matching_sizes.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/to_fvar.hpp>

namespace stan {
  namespace agrad {

    //dot_product for vec (in matrix) * vec (in matrix); does all combos of row row, col col, row col, col row
    template<typename T, int R1, int C1, int R2, int C2>
    inline 
    fvar<T> 
    dot_product(const Eigen::Matrix<fvar<T>, R1, C1>& v1, 
                const Eigen::Matrix<fvar<T>, R2, C2>& v2) {
      stan::error_handling::check_vector("dot_product", "v1", v1);
      stan::error_handling::check_vector("dot_product", "v2", v2);
      stan::error_handling::check_matching_sizes("dot_product",
                                                 "v1", v1,
                                                 "v2", v2);

      fvar<T> ret(0,0);
      for(size_type i = 0; i < v1.size(); i++)
        ret += v1(i) * v2(i);
      return ret;
    }

    template<typename T, int R1, int C1, int R2, int C2>
    inline 
    fvar<T> 
    dot_product(const Eigen::Matrix<fvar<T>, R1, C1>& v1, 
                const Eigen::Matrix<double, R2, C2>& v2) {
      stan::error_handling::check_vector("dot_product", "v1", v1);
      stan::error_handling::check_vector("dot_product", "v2", v2);
      stan::error_handling::check_matching_sizes("dot_product",
                                                 "v1", v1,
                                                 "v2", v2);

      fvar<T> ret(0,0);
      for(size_type i = 0; i < v1.size(); i++)
        ret += v1(i) * v2(i);
      return ret;
    }

    template<typename T, int R1, int C1, int R2, int C2>
    inline 
    fvar<T> 
    dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                const Eigen::Matrix<fvar<T>, R2, C2>& v2) {
      stan::error_handling::check_vector("dot_product", "v1", v1);
      stan::error_handling::check_vector("dot_product", "v2", v2);
      stan::error_handling::check_matching_sizes("dot_product",
                                                 "v1", v1,
                                                 "v2", v2);

      fvar<T> ret(0,0);
      for(size_type i = 0; i < v1.size(); i++)
        ret += v1(i) * v2(i);
      return ret;
    }

    template<typename T, int R1,int C1, int R2, int C2>
    inline 
    fvar<T> 
    dot_product(const Eigen::Matrix<fvar<T>, R1, C1>& v1, 
                const Eigen::Matrix<fvar<T>, R2, C2>& v2,
                size_type& length) {
      stan::error_handling::check_vector("dot_product", "v1", v1);
      stan::error_handling::check_vector("dot_product", "v2", v2);

      fvar<T> ret(0,0);
      for(size_type i = 0; i < length; i++)
        ret += v1(i) * v2(i);
      return ret;
    }

    template<typename T, int R1,int C1, int R2, int C2>
    inline 
    fvar<T> 
    dot_product(const Eigen::Matrix<fvar<T>, R1, C1>& v1, 
                const Eigen::Matrix<double, R2, C2>& v2,
                size_type& length) {
      stan::error_handling::check_vector("dot_product", "v1", v1);
      stan::error_handling::check_vector("dot_product", "v2", v2);

      fvar<T> ret(0,0);
      for(size_type i = 0; i < length; i++)
        ret += v1(i) * v2(i);
      return ret;
    }

    template<typename T, int R1,int C1, int R2, int C2>
    inline 
    fvar<T> 
    dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                const Eigen::Matrix<fvar<T>, R2, C2>& v2,
                size_type& length) {
      stan::error_handling::check_vector("dot_product", "v1", v1);
      stan::error_handling::check_vector("dot_product", "v2", v2);

      fvar<T> ret(0,0);
      for(size_type i = 0; i < length; i++)
        ret += v1(i) * v2(i);
      return ret;
    }

    template<typename T>
    inline 
    fvar<T>
    dot_product(const std::vector<fvar<T> >& v1,
                const std::vector<fvar<T> >& v2) {
      stan::error_handling::check_matching_sizes("dot_product",
                                                 "v1", v1,
                                                 "v2", v2);
      fvar<T> ret(0,0);
      for (size_t i = 0; i < v1.size(); i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }

    template<typename T>
    inline 
    fvar<T>
    dot_product(const std::vector<double>& v1,
                const std::vector<fvar<T> >& v2) {
      stan::error_handling::check_matching_sizes("dot_product",
                                                 "v1", v1,
                                                 "v2", v2);
      fvar<T> ret(0,0);
      for (size_t i = 0; i < v1.size(); i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }

    template<typename T>
    inline 
    fvar<T>
    dot_product(const std::vector<fvar<T> >& v1,
                const std::vector<double>& v2) {
      stan::error_handling::check_matching_sizes("dot_product",
                                                 "v1", v1,
                                                 "v2", v2);
      fvar<T> ret(0,0);
      for (size_t i = 0; i < v1.size(); i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }

    template<typename T>
    inline 
    fvar<T>
    dot_product(const std::vector<fvar<T> >& v1,
                const std::vector<fvar<T> >& v2,
                size_type& length) {
      fvar<T> ret(0,0);
      for(size_type i = 0; i < length; i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }

    template<typename T>
    inline 
    fvar<T>
    dot_product(const std::vector<double>& v1,
                const std::vector<fvar<T> >& v2,
                size_type& length) {
      fvar<T> ret(0,0);
      for(size_type i = 0; i < length; i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }

    template<typename T>
    inline 
    fvar<T>
    dot_product(const std::vector<fvar<T> >& v1,
                const std::vector<double>& v2,
                size_type& length) {
      fvar<T> ret(0,0);
      for(size_type i = 0; i < length; i++)
        ret += v1.at(i) * v2.at(i);
      return ret;
    }
  }
}
#endif
