#include <stan/math/fwd/mat/fun/to_fvar.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>

TEST(AgradMixMatrixToFvar,fv_vector) {
  using stan::math::vector_d;
  using stan::math::vector_fv;

  vector_fv v(5);
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;

  vector_fv out = stan::math::to_fvar(v);
  EXPECT_FLOAT_EQ(1, out(0).val_.val());
  EXPECT_FLOAT_EQ(2, out(1).val_.val());
  EXPECT_FLOAT_EQ(3, out(2).val_.val());
  EXPECT_FLOAT_EQ(4, out(3).val_.val());
  EXPECT_FLOAT_EQ(5, out(4).val_.val());  
  EXPECT_FLOAT_EQ(1, out(0).d_.val());
  EXPECT_FLOAT_EQ(1, out(1).d_.val());
  EXPECT_FLOAT_EQ(1, out(2).d_.val());
  EXPECT_FLOAT_EQ(1, out(3).d_.val());
  EXPECT_FLOAT_EQ(1, out(4).d_.val());
}
TEST(AgradMixMatrixToFvar,fv_rowvector) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_fv v(5);
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;

  row_vector_fv output = stan::math::to_fvar(v);
  EXPECT_FLOAT_EQ(1, output(0).val_.val());
  EXPECT_FLOAT_EQ(2, output(1).val_.val());
  EXPECT_FLOAT_EQ(3, output(2).val_.val());
  EXPECT_FLOAT_EQ(4, output(3).val_.val());
  EXPECT_FLOAT_EQ(5, output(4).val_.val());
  EXPECT_FLOAT_EQ(1, output(0).d_.val());
  EXPECT_FLOAT_EQ(1, output(1).d_.val());
  EXPECT_FLOAT_EQ(1, output(2).d_.val());
  EXPECT_FLOAT_EQ(1, output(3).d_.val());
  EXPECT_FLOAT_EQ(1, output(4).d_.val());
}
TEST(AgradMixMatrixToFvar,fv_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::matrix_fv;
  using stan::math::var;

  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> val(3,3);
  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> d(3,3);
  
  val <<1,2,3,4,5,6,7,8,9;
  d <<10,11,12,13,14,15,16,17,18;
  
  matrix_fv output = stan::math::to_fvar(val,d);
  EXPECT_FLOAT_EQ(1, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(2, output(0,1).val_.val());
  EXPECT_FLOAT_EQ(3, output(0,2).val_.val());
  EXPECT_FLOAT_EQ(4, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(5, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(6, output(1,2).val_.val());
  EXPECT_FLOAT_EQ(7, output(2,0).val_.val());
  EXPECT_FLOAT_EQ(8, output(2,1).val_.val());
  EXPECT_FLOAT_EQ(9, output(2,2).val_.val());
  EXPECT_FLOAT_EQ(10, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(11, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(12, output(0,2).d_.val());
  EXPECT_FLOAT_EQ(13, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(14, output(1,1).d_.val());
  EXPECT_FLOAT_EQ(15, output(1,2).d_.val());
  EXPECT_FLOAT_EQ(16, output(2,0).d_.val());
  EXPECT_FLOAT_EQ(17, output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18, output(2,2).d_.val());
}
TEST(AgradMixMatrixToFvar,ffv_vector) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_ffv v(5);
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;

  vector_ffv out = stan::math::to_fvar(v);
  EXPECT_FLOAT_EQ(1, out(0).val_.val().val());
  EXPECT_FLOAT_EQ(2, out(1).val_.val().val());
  EXPECT_FLOAT_EQ(3, out(2).val_.val().val());
  EXPECT_FLOAT_EQ(4, out(3).val_.val().val());
  EXPECT_FLOAT_EQ(5, out(4).val_.val().val());  
  EXPECT_FLOAT_EQ(1, out(0).d_.val().val());
  EXPECT_FLOAT_EQ(1, out(1).d_.val().val());
  EXPECT_FLOAT_EQ(1, out(2).d_.val().val());
  EXPECT_FLOAT_EQ(1, out(3).d_.val().val());
  EXPECT_FLOAT_EQ(1, out(4).d_.val().val());
}
TEST(AgradMixMatrixToFvar,ffv_rowvector) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_ffv v(5);
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;

  row_vector_ffv output = stan::math::to_fvar(v);
  EXPECT_FLOAT_EQ(1, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(2, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(3, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(4, output(3).val_.val().val());
  EXPECT_FLOAT_EQ(5, output(4).val_.val().val());
  EXPECT_FLOAT_EQ(1, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(1, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(1, output(2).d_.val().val());
  EXPECT_FLOAT_EQ(1, output(3).d_.val().val());
  EXPECT_FLOAT_EQ(1, output(4).d_.val().val());
}
TEST(AgradMixMatrixToFvar,ffv_matrix_matrix) {
  using stan::math::matrix_fv;
  using stan::math::matrix_ffv;

  matrix_fv val(3,3);
  matrix_fv d(3,3);
  
  val <<1,2,3,4,5,6,7,8,9;
  d <<10,11,12,13,14,15,16,17,18;
  
  matrix_ffv output = stan::math::to_fvar(val,d);
  EXPECT_FLOAT_EQ(1, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(2, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(3, output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ(4, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(5, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(6, output(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(7, output(2,0).val_.val().val());
  EXPECT_FLOAT_EQ(8, output(2,1).val_.val().val());
  EXPECT_FLOAT_EQ(9, output(2,2).val_.val().val());
  EXPECT_FLOAT_EQ(10, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(11, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(12, output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ(13, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(14, output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(15, output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(16, output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(17, output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(18, output(2,2).d_.val().val());
}
