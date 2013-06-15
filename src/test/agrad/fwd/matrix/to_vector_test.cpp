#include <stan/math/matrix/to_vector.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrix, to_vector) {
  using stan::math::to_vector;

  stan::agrad::matrix_fv a(3,3);
  stan::agrad::vector_fv b(9);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      a(j,i).val_ = j + i;
      a(j,i).d_ = j;
    }
  }

  b = to_vector(a);

  EXPECT_FLOAT_EQ(0,b(0).val());
  EXPECT_FLOAT_EQ(1,b(1).val());
  EXPECT_FLOAT_EQ(2,b(2).val());
  EXPECT_FLOAT_EQ(1,b(3).val());
  EXPECT_FLOAT_EQ(2,b(4).val());
  EXPECT_FLOAT_EQ(3,b(5).val());
  EXPECT_FLOAT_EQ(2,b(6).val());
  EXPECT_FLOAT_EQ(3,b(7).val());
  EXPECT_FLOAT_EQ(4,b(8).val());
  EXPECT_FLOAT_EQ(0,b(0).d_);
  EXPECT_FLOAT_EQ(1,b(1).d_);
  EXPECT_FLOAT_EQ(2,b(2).d_);
  EXPECT_FLOAT_EQ(0,b(3).d_);
  EXPECT_FLOAT_EQ(1,b(4).d_);
  EXPECT_FLOAT_EQ(2,b(5).d_);
  EXPECT_FLOAT_EQ(0,b(6).d_);
  EXPECT_FLOAT_EQ(1,b(7).d_);
  EXPECT_FLOAT_EQ(2,b(8).d_);
}
TEST(AgradFwdFvarVarMatrix, to_vector) {
  using stan::math::to_vector;
  using stan::agrad::var;
  using stan::agrad::fvar;

  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> a(3,3);
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,1> b(9);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      a(j,i).val_ = j + i;
      a(j,i).d_ = j;
    }
  }

  b = to_vector(a);

  EXPECT_FLOAT_EQ(0,b(0).val().val());
  EXPECT_FLOAT_EQ(1,b(1).val().val());
  EXPECT_FLOAT_EQ(2,b(2).val().val());
  EXPECT_FLOAT_EQ(1,b(3).val().val());
  EXPECT_FLOAT_EQ(2,b(4).val().val());
  EXPECT_FLOAT_EQ(3,b(5).val().val());
  EXPECT_FLOAT_EQ(2,b(6).val().val());
  EXPECT_FLOAT_EQ(3,b(7).val().val());
  EXPECT_FLOAT_EQ(4,b(8).val().val());
  EXPECT_FLOAT_EQ(0,b(0).d_.val());
  EXPECT_FLOAT_EQ(1,b(1).d_.val());
  EXPECT_FLOAT_EQ(2,b(2).d_.val());
  EXPECT_FLOAT_EQ(0,b(3).d_.val());
  EXPECT_FLOAT_EQ(1,b(4).d_.val());
  EXPECT_FLOAT_EQ(2,b(5).d_.val());
  EXPECT_FLOAT_EQ(0,b(6).d_.val());
  EXPECT_FLOAT_EQ(1,b(7).d_.val());
  EXPECT_FLOAT_EQ(2,b(8).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, to_vector) {
  using stan::math::to_vector;
  using stan::agrad::fvar;

  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic> a(3,3);
  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,1> b(9);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      a(j,i).val_ = j + i;
      a(j,i).d_ = j;
    }
  }

  b = to_vector(a);

  EXPECT_FLOAT_EQ(0,b(0).val().val());
  EXPECT_FLOAT_EQ(1,b(1).val().val());
  EXPECT_FLOAT_EQ(2,b(2).val().val());
  EXPECT_FLOAT_EQ(1,b(3).val().val());
  EXPECT_FLOAT_EQ(2,b(4).val().val());
  EXPECT_FLOAT_EQ(3,b(5).val().val());
  EXPECT_FLOAT_EQ(2,b(6).val().val());
  EXPECT_FLOAT_EQ(3,b(7).val().val());
  EXPECT_FLOAT_EQ(4,b(8).val().val());
  EXPECT_FLOAT_EQ(0,b(0).d_.val());
  EXPECT_FLOAT_EQ(1,b(1).d_.val());
  EXPECT_FLOAT_EQ(2,b(2).d_.val());
  EXPECT_FLOAT_EQ(0,b(3).d_.val());
  EXPECT_FLOAT_EQ(1,b(4).d_.val());
  EXPECT_FLOAT_EQ(2,b(5).d_.val());
  EXPECT_FLOAT_EQ(0,b(6).d_.val());
  EXPECT_FLOAT_EQ(1,b(7).d_.val());
  EXPECT_FLOAT_EQ(2,b(8).d_.val());
}
