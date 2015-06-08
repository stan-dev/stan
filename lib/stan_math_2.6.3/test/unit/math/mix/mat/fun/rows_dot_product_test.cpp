#include <stan/math/fwd/mat/fun/rows_dot_product.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradMixMatrixRowsDotProduct,fv_vector_vector_1stDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_fv;

  vector_d vd_1(3), vd_2(3);
  vector_fv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
  
  vector_fv output(3);
  output = rows_dot_product(vv_1, vd_2);
  EXPECT_FLOAT_EQ( 4, output(0).val_.val());
  EXPECT_FLOAT_EQ(-6, output(1).val_.val());
  EXPECT_FLOAT_EQ( 5, output(2).val_.val());
  EXPECT_FLOAT_EQ( 4, output(0).d_.val());
  EXPECT_FLOAT_EQ(-2, output(1).d_.val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val());

  output = rows_dot_product(vd_1, vv_2);
  EXPECT_FLOAT_EQ( 4, output(0).val_.val());
  EXPECT_FLOAT_EQ(-6, output(1).val_.val());
  EXPECT_FLOAT_EQ( 5, output(2).val_.val());
  EXPECT_FLOAT_EQ( 1, output(0).d_.val());
  EXPECT_FLOAT_EQ( 3, output(1).d_.val());
  EXPECT_FLOAT_EQ(-5, output(2).d_.val());

  output = rows_dot_product(vv_1, vv_2);
  EXPECT_FLOAT_EQ( 4, output(0).val_.val());
  EXPECT_FLOAT_EQ(-6, output(1).val_.val());
  EXPECT_FLOAT_EQ( 5, output(2).val_.val());
  EXPECT_FLOAT_EQ( 5, output(0).d_.val());
  EXPECT_FLOAT_EQ( 1, output(1).d_.val());
  EXPECT_FLOAT_EQ(-6, output(2).d_.val());

  AVEC q = createAVEC(vv_1(0).val(),vv_1(1).val(),vv_1(2).val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,fv_vector_vector_2ndDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_fv;

  vector_d vd_1(3), vd_2(3);
  vector_fv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
  
  vector_fv output(3);

  output = rows_dot_product(vv_1, vv_2);
  AVEC q = createAVEC(vv_1(0).val(),vv_1(1).val(),vv_1(2).val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,fv_vector_vector_exception) {
  using stan::math::vector_d;
  using stan::math::vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  vector_d d2(2);
  vector_fv v2(4);

  EXPECT_THROW(rows_dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixRowsDotProduct,fv_rowvector_vector) {
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  vector_d d2(3);
  vector_fv v2(3);

  EXPECT_THROW(rows_dot_product(v1,d2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v2), std::invalid_argument);
} 
TEST(AgradMixMatrixRowsDotProduct,fv_vector_rowvector) {
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  row_vector_d d2(3);
  row_vector_fv v2(3);

  EXPECT_THROW(rows_dot_product(v1,d2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v2), std::invalid_argument);
} 
TEST(AgradMixMatrixRowsDotProduct,fv_rowvector_rowvector_1stDeriv) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_d d1(3), d2(3);
  row_vector_fv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;

  row_vector_fv output;
  output = rows_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 1, output(0).d_.val());

  output = rows_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ(-1, output(0).d_.val());

  output = rows_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 0, output(0).d_.val());

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(-2,h[1]);
  EXPECT_FLOAT_EQ(-1,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,fv_rowvector_rowvector_2ndDeriv) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_d d1(3), d2(3);
  row_vector_fv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;

  row_vector_fv output;
  output = rows_dot_product(v1, v2);

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,fv_matrix_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;

  matrix_d d1(3,3), d2(3,3);
  matrix_fv v1(3,3), v2(3,3);
  
  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
   v1(1,0).d_ = 1.0;
   v1(1,1).d_ = 1.0;
   v1(1,2).d_ = 1.0;
   v1(2,0).d_ = 1.0;
   v1(2,1).d_ = 1.0;
   v1(2,2).d_ = 1.0;
  d2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
  v2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
   v2(0,0).d_ = 1.0;
   v2(0,1).d_ = 1.0;
   v2(0,2).d_ = 1.0;
   v2(1,0).d_ = 1.0;
   v2(1,1).d_ = 1.0;
   v2(1,2).d_ = 1.0;
   v2(2,0).d_ = 1.0;
   v2(2,1).d_ = 1.0;
   v2(2,2).d_ = 1.0;

  vector_fv output;
  output = rows_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val());
  EXPECT_FLOAT_EQ( 1, output(0).d_.val());
  EXPECT_FLOAT_EQ( 1, output(1).d_.val());
  EXPECT_FLOAT_EQ( 1, output(2).d_.val());

  output = rows_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val());
  EXPECT_FLOAT_EQ(-1, output(0).d_.val());
  EXPECT_FLOAT_EQ(-1, output(1).d_.val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val());

  output = rows_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val());
  EXPECT_FLOAT_EQ( 0, output(0).d_.val());
  EXPECT_FLOAT_EQ( 0, output(1).d_.val());
  EXPECT_FLOAT_EQ( 0, output(2).d_.val());

  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(1,0).val(),v1(1,1).val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(-2,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixRowsDotProduct,fv_matrix_matrix_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;

  matrix_d d1(3,3), d2(3,3);
  matrix_fv v1(3,3), v2(3,3);
  
  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
   v1(1,0).d_ = 1.0;
   v1(1,1).d_ = 1.0;
   v1(1,2).d_ = 1.0;
   v1(2,0).d_ = 1.0;
   v1(2,1).d_ = 1.0;
   v1(2,2).d_ = 1.0;
  d2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
  v2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
   v2(0,0).d_ = 1.0;
   v2(0,1).d_ = 1.0;
   v2(0,2).d_ = 1.0;
   v2(1,0).d_ = 1.0;
   v2(1,1).d_ = 1.0;
   v2(1,2).d_ = 1.0;
   v2(2,0).d_ = 1.0;
   v2(2,1).d_ = 1.0;
   v2(2,2).d_ = 1.0;

  vector_fv output;
  output = rows_dot_product(v1, v2);

  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(1,0).val(),v1(1,1).val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixRowsDotProduct,fv_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::rows_dot_product;

  matrix_d d1(3,3);
  matrix_d d2(3,2);
  matrix_d d3(2,3);
  matrix_fv v1(3,3);
  matrix_fv v2(3,3);
  matrix_fv v3(3,2);
  matrix_fv v4(3,2);
  matrix_fv v5(2,3);
  matrix_fv v6(2,3);

  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v2 << 4, -2, -1, 2, 1, 2, 1, 3, -5;
  v3 << 4, -2, -1, 2, 1, 2;
  v4 << 4, -2, -1, 2, 1, 2;
  v5 << 4, -2, -1, 2, 1, 2;
  v6 << 4, -2, -1, 2, 1, 2;

  EXPECT_THROW(rows_dot_product(v1,d2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,d3), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v3), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v4), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v5), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v6), std::invalid_argument);

  EXPECT_THROW(rows_dot_product(d1,v3), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v4), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v5), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v6), std::invalid_argument);

  EXPECT_THROW(rows_dot_product(d2,v1), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d2,v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d2,v5), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d2,v6), std::invalid_argument);

  EXPECT_THROW(rows_dot_product(d3,v1), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d3,v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d3,v3), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d3,v4), std::invalid_argument);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_vector_vector_1stDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d vd_1(3), vd_2(3);
  vector_ffv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
  
  vector_ffv output(3);
  output = rows_dot_product(vv_1, vd_2);
  EXPECT_FLOAT_EQ( 4, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(-6, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 4, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(-2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val().val());

  output = rows_dot_product(vd_1, vv_2);
  EXPECT_FLOAT_EQ( 4, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(-6, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 1, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( 3, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(-5, output(2).d_.val().val());

  output = rows_dot_product(vv_1, vv_2);
  EXPECT_FLOAT_EQ( 4, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(-6, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 5, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( 1, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(-6, output(2).d_.val().val());

  AVEC q = createAVEC(vv_1(0).val().val(),vv_1(1).val().val(),vv_1(2).val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_vector_vector_2ndDeriv_1) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d vd_1(3), vd_2(3);
  vector_ffv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
  
  vector_ffv output(3);

  output = rows_dot_product(vv_1, vv_2);
  AVEC q = createAVEC(vv_1(0).val().val(),vv_1(1).val().val(),vv_1(2).val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_vector_vector_2ndDeriv_2) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d vd_1(3), vd_2(3);
  vector_ffv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
  
  vector_ffv output(3);

  output = rows_dot_product(vv_1, vv_2);
  AVEC q = createAVEC(vv_1(0).val().val(),vv_1(1).val().val(),vv_1(2).val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_vector_vector_3rdDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d vd_1(3), vd_2(3);
  vector_ffv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;

   vv_1(0).val_.d_ = 1.0;
   vv_1(1).val_.d_ = 1.0;
   vv_1(2).val_.d_ = 1.0;
   vv_2(0).val_.d_ = 1.0;
   vv_2(1).val_.d_ = 1.0;
   vv_2(2).val_.d_ = 1.0;
  
  vector_ffv output(3);

  output = rows_dot_product(vv_1, vv_2);
  AVEC q = createAVEC(vv_1(0).val().val(),vv_1(1).val().val(),vv_1(2).val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_vector_vector_exception) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d d1(3);
  vector_ffv v1(3);
  vector_d d2(2);
  vector_ffv v2(4);

  EXPECT_THROW(rows_dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_rowvector_vector) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  vector_d d2(3);
  vector_ffv v2(3);

  EXPECT_THROW(rows_dot_product(v1,d2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v2), std::invalid_argument);
} 
TEST(AgradMixMatrixRowsDotProduct,ffv_vector_rowvector) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  vector_d d1(3);
  vector_ffv v1(3);
  row_vector_d d2(3);
  row_vector_ffv v2(3);

  EXPECT_THROW(rows_dot_product(v1,d2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v2), std::invalid_argument);
} 
TEST(AgradMixMatrixRowsDotProduct,ffv_rowvector_rowvector_1stDeriv) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3), d2(3);
  row_vector_ffv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;

  row_vector_ffv output;
  output = rows_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val().val());
  EXPECT_FLOAT_EQ( 1, output(0).d_.val().val());

  output = rows_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(-1, output(0).d_.val().val());

  output = rows_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val().val());
  EXPECT_FLOAT_EQ( 0, output(0).d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(-2,h[1]);
  EXPECT_FLOAT_EQ(-1,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_rowvector_rowvector_2ndDeriv_1) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3), d2(3);
  row_vector_ffv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;

  row_vector_ffv output;
  output = rows_dot_product(v1, v2);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_rowvector_rowvector_2ndDeriv_2) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3), d2(3);
  row_vector_ffv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;

  row_vector_ffv output;
  output = rows_dot_product(v1, v2);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(1,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_rowvector_rowvector_3rdDeriv) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3), d2(3);
  row_vector_ffv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;

   v1(0).val_.d_ = 1.0;
   v1(1).val_.d_ = 1.0;
   v1(2).val_.d_ = 1.0;
   v2(0).val_.d_ = 1.0;
   v2(1).val_.d_ = 1.0;
   v2(2).val_.d_ = 1.0;

  row_vector_ffv output;
  output = rows_dot_product(v1, v2);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_matrix_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_d d1(3,3), d2(3,3);
  matrix_ffv v1(3,3), v2(3,3);
  
  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
   v1(1,0).d_ = 1.0;
   v1(1,1).d_ = 1.0;
   v1(1,2).d_ = 1.0;
   v1(2,0).d_ = 1.0;
   v1(2,1).d_ = 1.0;
   v1(2,2).d_ = 1.0;
  d2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
  v2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
   v2(0,0).d_ = 1.0;
   v2(0,1).d_ = 1.0;
   v2(0,2).d_ = 1.0;
   v2(1,0).d_ = 1.0;
   v2(1,1).d_ = 1.0;
   v2(1,2).d_ = 1.0;
   v2(2,0).d_ = 1.0;
   v2(2,1).d_ = 1.0;
   v2(2,2).d_ = 1.0;

  vector_ffv output;
  output = rows_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 1, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( 1, output(1).d_.val().val());
  EXPECT_FLOAT_EQ( 1, output(2).d_.val().val());

  output = rows_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(-1, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(-1, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val().val());

  output = rows_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 0, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( 0, output(1).d_.val().val());
  EXPECT_FLOAT_EQ( 0, output(2).d_.val().val());

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(-2,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_matrix_matrix_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_d d1(3,3), d2(3,3);
  matrix_ffv v1(3,3), v2(3,3);
  
  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
   v1(1,0).d_ = 1.0;
   v1(1,1).d_ = 1.0;
   v1(1,2).d_ = 1.0;
   v1(2,0).d_ = 1.0;
   v1(2,1).d_ = 1.0;
   v1(2,2).d_ = 1.0;
  d2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
  v2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
   v2(0,0).d_ = 1.0;
   v2(0,1).d_ = 1.0;
   v2(0,2).d_ = 1.0;
   v2(1,0).d_ = 1.0;
   v2(1,1).d_ = 1.0;
   v2(1,2).d_ = 1.0;
   v2(2,0).d_ = 1.0;
   v2(2,1).d_ = 1.0;
   v2(2,2).d_ = 1.0;

  vector_ffv output;
  output = rows_dot_product(v1, v2);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_matrix_matrix_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_d d1(3,3), d2(3,3);
  matrix_ffv v1(3,3), v2(3,3);
  
  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
   v1(1,0).d_ = 1.0;
   v1(1,1).d_ = 1.0;
   v1(1,2).d_ = 1.0;
   v1(2,0).d_ = 1.0;
   v1(2,1).d_ = 1.0;
   v1(2,2).d_ = 1.0;
  d2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
  v2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
   v2(0,0).d_ = 1.0;
   v2(0,1).d_ = 1.0;
   v2(0,2).d_ = 1.0;
   v2(1,0).d_ = 1.0;
   v2(1,1).d_ = 1.0;
   v2(1,2).d_ = 1.0;
   v2(2,0).d_ = 1.0;
   v2(2,1).d_ = 1.0;
   v2(2,2).d_ = 1.0;

  vector_ffv output;
  output = rows_dot_product(v1, v2);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_matrix_matrix_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_d d1(3,3), d2(3,3);
  matrix_ffv v1(3,3), v2(3,3);
  
  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
   v1(1,0).d_ = 1.0;
   v1(1,1).d_ = 1.0;
   v1(1,2).d_ = 1.0;
   v1(2,0).d_ = 1.0;
   v1(2,1).d_ = 1.0;
   v1(2,2).d_ = 1.0;
   v1(0,0).val_.d_ = 1.0;
   v1(0,1).val_.d_ = 1.0;
   v1(0,2).val_.d_ = 1.0;
   v1(1,0).val_.d_ = 1.0;
   v1(1,1).val_.d_ = 1.0;
   v1(1,2).val_.d_ = 1.0;
   v1(2,0).val_.d_ = 1.0;
   v1(2,1).val_.d_ = 1.0;
   v1(2,2).val_.d_ = 1.0;
  d2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
  v2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
   v2(0,0).d_ = 1.0;
   v2(0,1).d_ = 1.0;
   v2(0,2).d_ = 1.0;
   v2(1,0).d_ = 1.0;
   v2(1,1).d_ = 1.0;
   v2(1,2).d_ = 1.0;
   v2(2,0).d_ = 1.0;
   v2(2,1).d_ = 1.0;
   v2(2,2).d_ = 1.0;
   v2(0,0).val_.d_ = 1.0;
   v2(0,1).val_.d_ = 1.0;
   v2(0,2).val_.d_ = 1.0;
   v2(1,0).val_.d_ = 1.0;
   v2(1,1).val_.d_ = 1.0;
   v2(1,2).val_.d_ = 1.0;
   v2(2,0).val_.d_ = 1.0;
   v2(2,1).val_.d_ = 1.0;
   v2(2,2).val_.d_ = 1.0;

  vector_ffv output;
  output = rows_dot_product(v1, v2);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixRowsDotProduct,ffv_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::rows_dot_product;

  matrix_d d1(3,3);
  matrix_d d2(3,2);
  matrix_d d3(2,3);
  matrix_ffv v1(3,3);
  matrix_ffv v2(3,3);
  matrix_ffv v3(3,2);
  matrix_ffv v4(3,2);
  matrix_ffv v5(2,3);
  matrix_ffv v6(2,3);

  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v2 << 4, -2, -1, 2, 1, 2, 1, 3, -5;
  v3 << 4, -2, -1, 2, 1, 2;
  v4 << 4, -2, -1, 2, 1, 2;
  v5 << 4, -2, -1, 2, 1, 2;
  v6 << 4, -2, -1, 2, 1, 2;

  EXPECT_THROW(rows_dot_product(v1,d2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,d3), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v3), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v4), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v5), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(v1,v6), std::invalid_argument);

  EXPECT_THROW(rows_dot_product(d1,v3), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v4), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v5), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d1,v6), std::invalid_argument);

  EXPECT_THROW(rows_dot_product(d2,v1), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d2,v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d2,v5), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d2,v6), std::invalid_argument);

  EXPECT_THROW(rows_dot_product(d3,v1), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d3,v2), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d3,v3), std::invalid_argument);
  EXPECT_THROW(rows_dot_product(d3,v4), std::invalid_argument);
}
