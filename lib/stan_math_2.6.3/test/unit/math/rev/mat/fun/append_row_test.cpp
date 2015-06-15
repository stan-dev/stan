#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/append_row.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

using stan::math::sum;
using stan::math::append_row;
using stan::math::matrix_v;
using stan::math::vector_v;
using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(AgradRevMatrix, append_row_matrix) {
  matrix_v a(2,2);
  matrix_v a_exp(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  AVEC x;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      x.push_back(a(i,j));
      a_exp(i, j) = stan::math::exp(a(i, j));
    }
  }
  
  AVAR append_row_ab = sum(append_row(a_exp, b));

  VEC g = cgradvec(append_row_ab, x);
  
  size_t idx = 0;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(std::exp(a(i, j).val()), g[idx++]);
}

TEST(AgradRevMatrix, append_row_row_vector) {
  vector_v a(3);
  vector_v a_exp(3);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0;
       
  b << 4.0, 3.0, 0.0;

  AVEC x;
  for (int i = 0; i < 3; ++i) {
    x.push_back(a(i));
    a_exp(i) = stan::math::exp(a(i));
  }
  
  AVAR append_row_ab = sum(append_row(a_exp, b));

  VEC g = cgradvec(append_row_ab, x);
  
  size_t idx = 0;
  for (int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(std::exp(a(i).val()), g[idx++]);
}

template <typename T, int R, int C>
void correct_type_vector(const Eigen::Matrix<T,R,C>& x) {
  EXPECT_EQ(Eigen::Dynamic, R);
  EXPECT_EQ(1, C);
}

template <typename T, int R, int C>
void correct_type_matrix(const Eigen::Matrix<T,R,C>& x) {
  EXPECT_EQ(Eigen::Dynamic, C);
  EXPECT_EQ(Eigen::Dynamic, R);
}

TEST(MathMatrix, append_row_different_types) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using Eigen::RowVectorXd;
  using stan::math::matrix_v;
  using stan::math::row_vector_v;
  using stan::math::vector_v;
  using std::vector;  
  
  MatrixXd m33(3, 3);
  m33 << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
        
  MatrixXd m32(3, 2);
  m32 << 11, 12,
         13, 14,
         15, 16;

  MatrixXd m32b(2, 3);
  m32b = m32*1.101; //ensure some different values

  MatrixXd m23(2, 3);
  m23 << 21, 22, 23,
         24, 25, 26;
         
  VectorXd v3(3);
  v3 << 31, 
        32,
        33;
        
  VectorXd v3b(3);
  v3b << 34, 
         35,
         36;

  RowVectorXd rv3(3);
  rv3 << 41, 42, 43;
  
  RowVectorXd rv3b(3);
  rv3b << 44, 45, 46;
  
  MatrixXd vm33(3, 3);
  vm33 << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
        
  MatrixXd vm32(3, 2);
  vm32 << 11, 12,
         13, 14,
         15, 16;

  MatrixXd vm32b(2, 3);
  vm32b = vm32*1.101; //ensure some different values

  MatrixXd vm23(2, 3);
  vm23 << 21, 22, 23,
         24, 25, 26;
         
  VectorXd vv3(3);
  vv3 << 31, 
        32,
        33;
        
  VectorXd vv3b(3);
  vv3b << 34, 
         35,
         36;

  RowVectorXd vrv3(3);
  vrv3 << 41, 42, 43;
  
  RowVectorXd vrv3b(3);
  vrv3b << 44, 45, 46;

  correct_type_matrix(append_row(m23, vm33));
  correct_type_matrix(append_row(m33, vm23));
  correct_type_matrix(append_row(m32, vm32b));
  correct_type_matrix(append_row(m32b, vm32));
  correct_type_matrix(append_row(m33, vrv3));
  correct_type_matrix(append_row(rv3, vm33));
  correct_type_matrix(append_row(m23, vrv3));
  correct_type_matrix(append_row(rv3, vm23));
  correct_type_matrix(append_row(rv3, vrv3b));
  correct_type_matrix(append_row(rv3b, vrv3));
  correct_type_vector(append_row(v3, vv3b));
  correct_type_vector(append_row(v3b, vv3));

  correct_type_matrix(append_row(vm23, m33));
  correct_type_matrix(append_row(vm33, m23));
  correct_type_matrix(append_row(vm32, m32b));
  correct_type_matrix(append_row(vm32b, m32));
  correct_type_matrix(append_row(vm33, rv3));
  correct_type_matrix(append_row(vrv3, m33));
  correct_type_matrix(append_row(vm23, rv3));
  correct_type_matrix(append_row(vrv3, m23));
  correct_type_matrix(append_row(vrv3, rv3b));
  correct_type_matrix(append_row(vrv3b, rv3));
  correct_type_vector(append_row(vv3, v3b));
  correct_type_vector(append_row(vv3b, v3));

  correct_type_matrix(append_row(vm23, vm33));
  correct_type_matrix(append_row(vm33, vm23));
  correct_type_matrix(append_row(vm32, vm32b));
  correct_type_matrix(append_row(vm32b, vm32));
  correct_type_matrix(append_row(vm33, vrv3));
  correct_type_matrix(append_row(vrv3, vm33));
  correct_type_matrix(append_row(vm23, vrv3));
  correct_type_matrix(append_row(vrv3, vm23));
  correct_type_matrix(append_row(vrv3, vrv3b));
  correct_type_matrix(append_row(vrv3b, vrv3));
  correct_type_vector(append_row(vv3, vv3b));
  correct_type_vector(append_row(vv3b, vv3));
}
