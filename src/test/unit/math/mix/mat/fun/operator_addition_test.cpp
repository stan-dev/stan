#include <stan/math/prim/mat/fun/add.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
using stan::math::fvar;
TEST(AgradMixMatrixOperatorAddition,fv_scalar_matrix_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::add;

  matrix_fv v(2,2);
  v << 1, 2, 3, 4;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
  matrix_fv result;

  result = add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val_.val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val_.val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val_.val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1,0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1,1).d_.val());

  result = add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val_.val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val_.val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val_.val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1,0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1,1).d_.val());

  AVEC q = createAVEC(v(0,0).val(),v(0,1).val(),v(1,0).val(),v(1,1).val());
  VEC h;
  result(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_scalar_matrix_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::add;

  matrix_fv v(2,2);
  v << 1, 2, 3, 4;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
  matrix_fv result;
  result = add(v,2.0);

  AVEC q = createAVEC(v(0,0).val(),v(0,1).val(),v(1,0).val(),v(1,1).val());
  VEC h;
  result(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_scalar_vector_1stDeriv) {
  using stan::math::vector_fv;
  using stan::math::add;

  vector_fv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  vector_fv result;

  result = add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0).val_.val());
  EXPECT_FLOAT_EQ(4.0,result(1).val_.val());
  EXPECT_FLOAT_EQ(5.0,result(2).val_.val());
  EXPECT_FLOAT_EQ(6.0,result(3).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(2).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val());

  result = add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0).val_.val());
  EXPECT_FLOAT_EQ(4.0,result(1).val_.val());
  EXPECT_FLOAT_EQ(5.0,result(2).val_.val());
  EXPECT_FLOAT_EQ(6.0,result(3).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(2).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val());

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val(),v(3).val());
  VEC h;
  result(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_scalar_vector_2ndDeriv) {
  using stan::math::vector_fv;
  using stan::math::add;

  vector_fv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  vector_fv result;
  result = add(v,2.0);

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val(),v(3).val());
  VEC h;
  result(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_scalar_rowvector_1stDeriv) {
  using stan::math::row_vector_fv;
  using stan::math::add;

  row_vector_fv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  row_vector_fv result;

  result = add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0).val_.val());
  EXPECT_FLOAT_EQ(4.0,result(1).val_.val());
  EXPECT_FLOAT_EQ(5.0,result(2).val_.val());
  EXPECT_FLOAT_EQ(6.0,result(3).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(2).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val());

  result = add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0).val_.val());
  EXPECT_FLOAT_EQ(4.0,result(1).val_.val());
  EXPECT_FLOAT_EQ(5.0,result(2).val_.val());
  EXPECT_FLOAT_EQ(6.0,result(3).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(2).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val());

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val(),v(3).val());
  VEC h;
  result(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_scalar_rowvector_2ndDeriv) {
  using stan::math::row_vector_fv;
  using stan::math::add;

  row_vector_fv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  row_vector_fv result;

  result = add(v,2.0);

  AVEC q = createAVEC(v(0).val(),v(1).val(),v(2).val(),v(3).val());
  VEC h;
  result(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_vector_vector_1stDeriv) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_fv;

  vector_d vd_1(5);
  vector_d vd_2(5);
  vector_fv vv_1(5);
  vector_fv vv_2(5);
  
  vd_1 << 1, 2, 3, 4, 5;
  vv_1 << 1, 2, 3, 4, 5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
   vv_1(3).d_ = 1.0;
   vv_1(4).d_ = 1.0;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
   vv_2(3).d_ = 1.0;
   vv_2(4).d_ = 1.0;
  
  vector_d expected_output(5);
  expected_output << 3, 5, 7, 9, 11;
  
  vector_d output_d;
  output_d = add(vd_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  vector_fv output_v = add(vv_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val()); 
  EXPECT_FLOAT_EQ(1.0, output_v(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(2).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(3).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(4).d_.val()); 

  output_v = add(vd_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val()); 
  EXPECT_FLOAT_EQ(1.0, output_v(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(2).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(3).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(4).d_.val());   

  output_v = add(vv_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val()); 
  EXPECT_FLOAT_EQ(2.0, output_v(0).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(1).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(2).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(3).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(4).d_.val());  

  AVEC q = createAVEC(vv_1(0).val(),vv_1(1).val(),vv_1(2).val(),vv_1(3).val());
  VEC h;
  output_v(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_vector_vector_2ndDeriv) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_fv;

  vector_d vd_1(5);
  vector_d vd_2(5);
  vector_fv vv_1(5);
  vector_fv vv_2(5);
  
  vd_1 << 1, 2, 3, 4, 5;
  vv_1 << 1, 2, 3, 4, 5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
   vv_1(3).d_ = 1.0;
   vv_1(4).d_ = 1.0;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
   vv_2(3).d_ = 1.0;
   vv_2(4).d_ = 1.0;
  
  vector_fv output_v;
  output_v = add(vv_1, vv_2);

  AVEC q = createAVEC(vv_1(0).val(),vv_1(1).val(),vv_1(2).val(),vv_1(3).val());
  VEC h;
  output_v(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_vector_vector_exception) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_fv;

  vector_d d1(5), d2(1);
  vector_fv v1(5), v2(1);
  
  EXPECT_THROW(add(d1, d2), std::invalid_argument);
  EXPECT_THROW(add(v1, d2), std::invalid_argument);
  EXPECT_THROW(add(d1, v2), std::invalid_argument);
  EXPECT_THROW(add(v1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorAddition,fv_rowvector_rowvector_1stDeriv) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_fv rvv_1(5), rvv_2(5);

  rvd_1 << 1, 2, 3, 4, 5;
  rvv_1 << 1, 2, 3, 4, 5;
   rvv_1(0).d_ = 1.0;
   rvv_1(1).d_ = 1.0;
   rvv_1(2).d_ = 1.0;
   rvv_1(3).d_ = 1.0;
   rvv_1(4).d_ = 1.0;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
   rvv_2(0).d_ = 1.0;
   rvv_2(1).d_ = 1.0;
   rvv_2(2).d_ = 1.0;
   rvv_2(3).d_ = 1.0;
   rvv_2(4).d_ = 1.0;
  
  row_vector_d expected_output(5);
  expected_output << 3, 5, 7, 9, 11;
  
  row_vector_d output_d = add(rvd_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  row_vector_fv output_v = add(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val());  
  EXPECT_FLOAT_EQ(1.0, output_v(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(2).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(3).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(4).d_.val());

  output_v = add(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val());  
  EXPECT_FLOAT_EQ(1.0, output_v(0).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(1).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(2).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(3).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(4).d_.val());

  output_v = add(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val());  
  EXPECT_FLOAT_EQ(2.0, output_v(0).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(1).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(2).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(3).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(4).d_.val());

  AVEC q = createAVEC(rvv_1(0).val(),rvv_1(1).val(),rvv_1(2).val(),rvv_1(3).val());
  VEC h;
  output_v(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_rowvector_rowvector_2ndDeriv) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_fv rvv_1(5), rvv_2(5);

  rvd_1 << 1, 2, 3, 4, 5;
  rvv_1 << 1, 2, 3, 4, 5;
   rvv_1(0).d_ = 1.0;
   rvv_1(1).d_ = 1.0;
   rvv_1(2).d_ = 1.0;
   rvv_1(3).d_ = 1.0;
   rvv_1(4).d_ = 1.0;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
   rvv_2(0).d_ = 1.0;
   rvv_2(1).d_ = 1.0;
   rvv_2(2).d_ = 1.0;
   rvv_2(3).d_ = 1.0;
   rvv_2(4).d_ = 1.0;
  
  row_vector_fv output_v = add(rvv_1, rvd_2);
  output_v = add(rvv_1, rvv_2);

  AVEC q = createAVEC(rvv_1(0).val(),rvv_1(1).val(),rvv_1(2).val(),rvv_1(3).val());
  VEC h;
  output_v(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_rowvector_rowvector_exception) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_d d1(5), d2(2);
  row_vector_fv v1(5), v2(2);

  row_vector_fv output;
  EXPECT_THROW( add(d1, d2), std::invalid_argument);
  EXPECT_THROW( add(d1, v2), std::invalid_argument);
  EXPECT_THROW( add(v1, d2), std::invalid_argument);
  EXPECT_THROW( add(v1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorAddition,fv_matrix_matrix_1stDeriv) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;

  matrix_d md_1(2,2), md_2(2,2);
  matrix_fv mv_1(2,2), mv_2(2,2);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
   mv_1(0,0).d_ = 1.0;
   mv_1(0,1).d_ = 1.0;
   mv_1(1,0).d_ = 1.0;
   mv_1(1,1).d_ = 1.0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
   mv_2(0,0).d_ = 1.0;
   mv_2(0,1).d_ = 1.0;
   mv_2(1,0).d_ = 1.0;
   mv_2(1,1).d_ = 1.0;
  
  matrix_d expected_output(2,2);
  expected_output << 0, -9, 11, 2;
  
  matrix_d output_d = add(md_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_d(0,0));
  EXPECT_FLOAT_EQ(expected_output(0,1), output_d(0,1));
  EXPECT_FLOAT_EQ(expected_output(1,0), output_d(1,0));
  EXPECT_FLOAT_EQ(expected_output(1,1), output_d(1,1));

  matrix_fv output_v = add(mv_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(1,0).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(1,1).d_.val());

  output_v = add(md_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(1,0).d_.val());
  EXPECT_FLOAT_EQ(1.0, output_v(1,1).d_.val());

  output_v = add(mv_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(0,0).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(0,1).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(1,0).d_.val());
  EXPECT_FLOAT_EQ(2.0, output_v(1,1).d_.val());

  AVEC q = createAVEC(mv_1(0,0).val(),mv_1(0,1).val(),mv_1(1,0).val(),mv_1(1,1).val());
  VEC h;
  output_v(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_matrix_matrix_2ndDeriv) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;

  matrix_d md_1(2,2), md_2(2,2);
  matrix_fv mv_1(2,2), mv_2(2,2);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
   mv_1(0,0).d_ = 1.0;
   mv_1(0,1).d_ = 1.0;
   mv_1(1,0).d_ = 1.0;
   mv_1(1,1).d_ = 1.0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
   mv_2(0,0).d_ = 1.0;
   mv_2(0,1).d_ = 1.0;
   mv_2(1,0).d_ = 1.0;
   mv_2(1,1).d_ = 1.0;
  
  matrix_fv output_v = add(mv_1, md_2);
  output_v = add(mv_1, mv_2);

  AVEC q = createAVEC(mv_1(0,0).val(),mv_1(0,1).val(),mv_1(1,0).val(),mv_1(1,1).val());
  VEC h;
  output_v(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,fv_matrix_matrix_exception) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  
  matrix_d d1(2,2), d2(1,2);
  matrix_fv v1(2,2), v2(1,2);

  EXPECT_THROW(add(d1, d2), std::invalid_argument);
  EXPECT_THROW(add(d1, v2), std::invalid_argument);
  EXPECT_THROW(add(v1, d2), std::invalid_argument);
  EXPECT_THROW(add(v1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_matrix_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::add;

  matrix_ffv v(2,2);
  v << 1, 2, 3, 4;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
  matrix_ffv result;

  result = add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(1,1).d_.val().val());

  result = add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(1,1).d_.val().val());

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(1,0).val().val(),v(1,1).val().val());
  VEC h;
  result(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_matrix_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::add;

  matrix_ffv v(2,2);
  v << 1, 2, 3, 4;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
  matrix_ffv result;
  result = add(v,2.0);

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(1,0).val().val(),v(1,1).val().val());
  VEC h;
  result(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_matrix_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::add;

  matrix_ffv v(2,2);
  v << 1, 2, 3, 4;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
  matrix_ffv result;
  result = add(v,2.0);

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(1,0).val().val(),v(1,1).val().val());
  VEC h;
  result(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_matrix_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::add;

  matrix_ffv v(2,2);
  v << 1, 2, 3, 4;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 1.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 1.0;
   v(0,0).val_.d_ = 1.0;
   v(0,1).val_.d_ = 1.0;
   v(1,0).val_.d_ = 1.0;
   v(1,1).val_.d_ = 1.0;
  matrix_ffv result;
  result = add(v,2.0);

  AVEC q = createAVEC(v(0,0).val().val(),v(0,1).val().val(),v(1,0).val().val(),v(1,1).val().val());
  VEC h;
  result(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_vector_1stDeriv) {
  using stan::math::vector_ffv;
  using stan::math::add;

  vector_ffv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  vector_ffv result;

  result = add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0).val_.val().val());
  EXPECT_FLOAT_EQ(4.0,result(1).val_.val().val());
  EXPECT_FLOAT_EQ(5.0,result(2).val_.val().val());
  EXPECT_FLOAT_EQ(6.0,result(3).val_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(2).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val().val());

  result = add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0).val_.val().val());
  EXPECT_FLOAT_EQ(4.0,result(1).val_.val().val());
  EXPECT_FLOAT_EQ(5.0,result(2).val_.val().val());
  EXPECT_FLOAT_EQ(6.0,result(3).val_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(2).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val().val());

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  result(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_vector_2ndDeriv_1) {
  using stan::math::vector_ffv;
  using stan::math::add;

  vector_ffv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  vector_ffv result;
  result = add(v,2.0);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  result(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_vector_2ndDeriv_2) {
  using stan::math::vector_ffv;
  using stan::math::add;

  vector_ffv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  vector_ffv result;
  result = add(v,2.0);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  result(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_vector_3rdDeriv) {
  using stan::math::vector_ffv;
  using stan::math::add;

  vector_ffv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(0).val_.d_ = 1.0;
   v(1).val_.d_ = 1.0;
   v(2).val_.d_ = 1.0;
   v(3).val_.d_ = 1.0;
  vector_ffv result;
  result = add(v,2.0);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  result(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_rowvector_1stDeriv) {
  using stan::math::row_vector_ffv;
  using stan::math::add;

  row_vector_ffv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  row_vector_ffv result;

  result = add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0).val_.val().val());
  EXPECT_FLOAT_EQ(4.0,result(1).val_.val().val());
  EXPECT_FLOAT_EQ(5.0,result(2).val_.val().val());
  EXPECT_FLOAT_EQ(6.0,result(3).val_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(2).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val().val());

  result = add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0).val_.val().val());
  EXPECT_FLOAT_EQ(4.0,result(1).val_.val().val());
  EXPECT_FLOAT_EQ(5.0,result(2).val_.val().val());
  EXPECT_FLOAT_EQ(6.0,result(3).val_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(2).d_.val().val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val().val());

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  result(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_rowvector_2ndDeriv_1) {
  using stan::math::row_vector_ffv;
  using stan::math::add;

  row_vector_ffv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  row_vector_ffv result;

  result = add(v,2.0);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  result(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_rowvector_2ndDeriv_2) {
  using stan::math::row_vector_ffv;
  using stan::math::add;

  row_vector_ffv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  row_vector_ffv result;

  result = add(v,2.0);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  result(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_scalar_rowvector_3rdDeriv) {
  using stan::math::row_vector_ffv;
  using stan::math::add;

  row_vector_ffv v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(0).val_.d_ = 1.0;
   v(1).val_.d_ = 1.0;
   v(2).val_.d_ = 1.0;
   v(3).val_.d_ = 1.0;
  row_vector_ffv result;

  result = add(v,2.0);

  AVEC q = createAVEC(v(0).val().val(),v(1).val().val(),v(2).val().val(),v(3).val().val());
  VEC h;
  result(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_vector_vector_1stDeriv) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d vd_1(5);
  vector_d vd_2(5);
  vector_ffv vv_1(5);
  vector_ffv vv_2(5);
  
  vd_1 << 1, 2, 3, 4, 5;
  vv_1 << 1, 2, 3, 4, 5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
   vv_1(3).d_ = 1.0;
   vv_1(4).d_ = 1.0;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
   vv_2(3).d_ = 1.0;
   vv_2(4).d_ = 1.0;
  
  vector_d expected_output(5);
  expected_output << 3, 5, 7, 9, 11;
  
  vector_d output_d;
  output_d = add(vd_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  vector_ffv output_v = add(vv_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val().val()); 
  EXPECT_FLOAT_EQ(1.0, output_v(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(2).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(3).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(4).d_.val().val()); 

  output_v = add(vd_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val().val()); 
  EXPECT_FLOAT_EQ(1.0, output_v(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(2).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(3).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(4).d_.val().val());   

  output_v = add(vv_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val().val()); 
  EXPECT_FLOAT_EQ(2.0, output_v(0).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(1).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(2).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(3).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(4).d_.val().val());  

  AVEC q = createAVEC(vv_1(0).val().val(),vv_1(1).val().val(),vv_1(2).val().val(),vv_1(3).val().val());
  VEC h;
  output_v(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_vector_vector_2ndDeriv_1) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d vd_1(5);
  vector_d vd_2(5);
  vector_ffv vv_1(5);
  vector_ffv vv_2(5);
  
  vd_1 << 1, 2, 3, 4, 5;
  vv_1 << 1, 2, 3, 4, 5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
   vv_1(3).d_ = 1.0;
   vv_1(4).d_ = 1.0;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
   vv_2(3).d_ = 1.0;
   vv_2(4).d_ = 1.0;
  
  vector_ffv output_v;
  output_v = add(vv_1, vv_2);

  AVEC q = createAVEC(vv_1(0).val().val(),vv_1(1).val().val(),vv_1(2).val().val(),vv_1(3).val().val());
  VEC h;
  output_v(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_vector_vector_2ndDeriv_2) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d vd_1(5);
  vector_d vd_2(5);
  vector_ffv vv_1(5);
  vector_ffv vv_2(5);
  
  vd_1 << 1, 2, 3, 4, 5;
  vv_1 << 1, 2, 3, 4, 5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
   vv_1(3).d_ = 1.0;
   vv_1(4).d_ = 1.0;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
   vv_2(3).d_ = 1.0;
   vv_2(4).d_ = 1.0;
  
  vector_ffv output_v;
  output_v = add(vv_1, vv_2);

  AVEC q = createAVEC(vv_1(0).val().val(),vv_1(1).val().val(),vv_1(2).val().val(),vv_1(3).val().val());
  VEC h;
  output_v(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_vector_vector_3rdDeriv) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d vd_1(5);
  vector_d vd_2(5);
  vector_ffv vv_1(5);
  vector_ffv vv_2(5);
  
  vd_1 << 1, 2, 3, 4, 5;
  vv_1 << 1, 2, 3, 4, 5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
   vv_1(3).d_ = 1.0;
   vv_1(4).d_ = 1.0;
   vv_1(0).val_.d_ = 1.0;
   vv_1(1).val_.d_ = 1.0;
   vv_1(2).val_.d_ = 1.0;
   vv_1(3).val_.d_ = 1.0;
   vv_1(4).val_.d_ = 1.0;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
   vv_2(3).d_ = 1.0;
   vv_2(4).d_ = 1.0;
   vv_2(0).val_.d_ = 1.0;
   vv_2(1).val_.d_ = 1.0;
   vv_2(2).val_.d_ = 1.0;
   vv_2(3).val_.d_ = 1.0;
   vv_2(4).val_.d_ = 1.0;
  
  vector_ffv output_v;
  output_v = add(vv_1, vv_2);

  AVEC q = createAVEC(vv_1(0).val().val(),vv_1(1).val().val(),vv_1(2).val().val(),vv_1(3).val().val());
  VEC h;
  output_v(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_vector_vector_exception) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d d1(5), d2(1);
  vector_ffv v1(5), v2(1);
  
  EXPECT_THROW(add(d1, d2), std::invalid_argument);
  EXPECT_THROW(add(v1, d2), std::invalid_argument);
  EXPECT_THROW(add(d1, v2), std::invalid_argument);
  EXPECT_THROW(add(v1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorAddition,ffv_rowvector_rowvector_1stDeriv) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_ffv rvv_1(5), rvv_2(5);

  rvd_1 << 1, 2, 3, 4, 5;
  rvv_1 << 1, 2, 3, 4, 5;
   rvv_1(0).d_ = 1.0;
   rvv_1(1).d_ = 1.0;
   rvv_1(2).d_ = 1.0;
   rvv_1(3).d_ = 1.0;
   rvv_1(4).d_ = 1.0;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
   rvv_2(0).d_ = 1.0;
   rvv_2(1).d_ = 1.0;
   rvv_2(2).d_ = 1.0;
   rvv_2(3).d_ = 1.0;
   rvv_2(4).d_ = 1.0;
  
  row_vector_d expected_output(5);
  expected_output << 3, 5, 7, 9, 11;
  
  row_vector_d output_d = add(rvd_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  row_vector_ffv output_v = add(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val().val());  
  EXPECT_FLOAT_EQ(1.0, output_v(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(2).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(3).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(4).d_.val().val());

  output_v = add(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val().val());  
  EXPECT_FLOAT_EQ(1.0, output_v(0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(2).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(3).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(4).d_.val().val());

  output_v = add(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val_.val().val());  
  EXPECT_FLOAT_EQ(2.0, output_v(0).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(1).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(2).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(3).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(4).d_.val().val());

  AVEC q = createAVEC(rvv_1(0).val().val(),rvv_1(1).val().val(),rvv_1(2).val().val(),rvv_1(3).val().val());
  VEC h;
  output_v(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_rowvector_rowvector_2ndDeriv_1) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_ffv rvv_1(5), rvv_2(5);

  rvd_1 << 1, 2, 3, 4, 5;
  rvv_1 << 1, 2, 3, 4, 5;
   rvv_1(0).d_ = 1.0;
   rvv_1(1).d_ = 1.0;
   rvv_1(2).d_ = 1.0;
   rvv_1(3).d_ = 1.0;
   rvv_1(4).d_ = 1.0;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
   rvv_2(0).d_ = 1.0;
   rvv_2(1).d_ = 1.0;
   rvv_2(2).d_ = 1.0;
   rvv_2(3).d_ = 1.0;
   rvv_2(4).d_ = 1.0;
  
  row_vector_ffv output_v = add(rvv_1, rvd_2);
  output_v = add(rvv_1, rvv_2);

  AVEC q = createAVEC(rvv_1(0).val().val(),rvv_1(1).val().val(),rvv_1(2).val().val(),rvv_1(3).val().val());
  VEC h;
  output_v(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_rowvector_rowvector_2ndDeriv_2) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_ffv rvv_1(5), rvv_2(5);

  rvd_1 << 1, 2, 3, 4, 5;
  rvv_1 << 1, 2, 3, 4, 5;
   rvv_1(0).d_ = 1.0;
   rvv_1(1).d_ = 1.0;
   rvv_1(2).d_ = 1.0;
   rvv_1(3).d_ = 1.0;
   rvv_1(4).d_ = 1.0;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
   rvv_2(0).d_ = 1.0;
   rvv_2(1).d_ = 1.0;
   rvv_2(2).d_ = 1.0;
   rvv_2(3).d_ = 1.0;
   rvv_2(4).d_ = 1.0;
  
  row_vector_ffv output_v = add(rvv_1, rvd_2);
  output_v = add(rvv_1, rvv_2);

  AVEC q = createAVEC(rvv_1(0).val().val(),rvv_1(1).val().val(),rvv_1(2).val().val(),rvv_1(3).val().val());
  VEC h;
  output_v(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_rowvector_rowvector_3rdDeriv) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_ffv rvv_1(5), rvv_2(5);

  rvd_1 << 1, 2, 3, 4, 5;
  rvv_1 << 1, 2, 3, 4, 5;
   rvv_1(0).d_ = 1.0;
   rvv_1(1).d_ = 1.0;
   rvv_1(2).d_ = 1.0;
   rvv_1(3).d_ = 1.0;
   rvv_1(4).d_ = 1.0;
   rvv_1(0).val_.d_ = 1.0;
   rvv_1(1).val_.d_ = 1.0;
   rvv_1(2).val_.d_ = 1.0;
   rvv_1(3).val_.d_ = 1.0;
   rvv_1(4).val_.d_ = 1.0;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
   rvv_2(0).d_ = 1.0;
   rvv_2(1).d_ = 1.0;
   rvv_2(2).d_ = 1.0;
   rvv_2(3).d_ = 1.0;
   rvv_2(4).d_ = 1.0;
   rvv_2(0).val_.d_ = 1.0;
   rvv_2(1).val_.d_ = 1.0;
   rvv_2(2).val_.d_ = 1.0;
   rvv_2(3).val_.d_ = 1.0;
   rvv_2(4).val_.d_ = 1.0;
  
  row_vector_ffv output_v = add(rvv_1, rvd_2);
  output_v = add(rvv_1, rvv_2);

  AVEC q = createAVEC(rvv_1(0).val().val(),rvv_1(1).val().val(),rvv_1(2).val().val(),rvv_1(3).val().val());
  VEC h;
  output_v(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_rowvector_rowvector_exception) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(5), d2(2);
  row_vector_ffv v1(5), v2(2);

  row_vector_ffv output;
  EXPECT_THROW( add(d1, d2), std::invalid_argument);
  EXPECT_THROW( add(d1, v2), std::invalid_argument);
  EXPECT_THROW( add(v1, d2), std::invalid_argument);
  EXPECT_THROW( add(v1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorAddition,ffv_matrix_matrix_1stDeriv) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d md_1(2,2), md_2(2,2);
  matrix_ffv mv_1(2,2), mv_2(2,2);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
   mv_1(0,0).d_ = 1.0;
   mv_1(0,1).d_ = 1.0;
   mv_1(1,0).d_ = 1.0;
   mv_1(1,1).d_ = 1.0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
   mv_2(0,0).d_ = 1.0;
   mv_2(0,1).d_ = 1.0;
   mv_2(1,0).d_ = 1.0;
   mv_2(1,1).d_ = 1.0;
  
  matrix_d expected_output(2,2);
  expected_output << 0, -9, 11, 2;
  
  matrix_d output_d = add(md_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_d(0,0));
  EXPECT_FLOAT_EQ(expected_output(0,1), output_d(0,1));
  EXPECT_FLOAT_EQ(expected_output(1,0), output_d(1,0));
  EXPECT_FLOAT_EQ(expected_output(1,1), output_d(1,1));

  matrix_ffv output_v = add(mv_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(1,1).d_.val().val());

  output_v = add(md_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(1.0, output_v(1,1).d_.val().val());

  output_v = add(mv_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(2.0, output_v(1,1).d_.val().val());

  AVEC q = createAVEC(mv_1(0,0).val().val(),mv_1(0,1).val().val(),mv_1(1,0).val().val(),mv_1(1,1).val().val());
  VEC h;
  output_v(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_matrix_matrix_2ndDeriv_1) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d md_1(2,2), md_2(2,2);
  matrix_ffv mv_1(2,2), mv_2(2,2);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
   mv_1(0,0).d_ = 1.0;
   mv_1(0,1).d_ = 1.0;
   mv_1(1,0).d_ = 1.0;
   mv_1(1,1).d_ = 1.0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
   mv_2(0,0).d_ = 1.0;
   mv_2(0,1).d_ = 1.0;
   mv_2(1,0).d_ = 1.0;
   mv_2(1,1).d_ = 1.0;
  
  matrix_ffv output_v = add(mv_1, md_2);
  output_v = add(mv_1, mv_2);

  AVEC q = createAVEC(mv_1(0,0).val().val(),mv_1(0,1).val().val(),mv_1(1,0).val().val(),mv_1(1,1).val().val());
  VEC h;
  output_v(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_matrix_matrix_2ndDeriv_2) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d md_1(2,2), md_2(2,2);
  matrix_ffv mv_1(2,2), mv_2(2,2);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
   mv_1(0,0).d_ = 1.0;
   mv_1(0,1).d_ = 1.0;
   mv_1(1,0).d_ = 1.0;
   mv_1(1,1).d_ = 1.0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
   mv_2(0,0).d_ = 1.0;
   mv_2(0,1).d_ = 1.0;
   mv_2(1,0).d_ = 1.0;
   mv_2(1,1).d_ = 1.0;
  
  matrix_ffv output_v = add(mv_1, md_2);
  output_v = add(mv_1, mv_2);

  AVEC q = createAVEC(mv_1(0,0).val().val(),mv_1(0,1).val().val(),mv_1(1,0).val().val(),mv_1(1,1).val().val());
  VEC h;
  output_v(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_matrix_matrix_3rdDeriv) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d md_1(2,2), md_2(2,2);
  matrix_ffv mv_1(2,2), mv_2(2,2);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
   mv_1(0,0).d_ = 1.0;
   mv_1(0,1).d_ = 1.0;
   mv_1(1,0).d_ = 1.0;
   mv_1(1,1).d_ = 1.0;
   mv_1(0,0).val_.d_ = 1.0;
   mv_1(0,1).val_.d_ = 1.0;
   mv_1(1,0).val_.d_ = 1.0;
   mv_1(1,1).val_.d_ = 1.0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
   mv_2(0,0).d_ = 1.0;
   mv_2(0,1).d_ = 1.0;
   mv_2(1,0).d_ = 1.0;
   mv_2(1,1).d_ = 1.0;
   mv_2(0,0).val_.d_ = 1.0;
   mv_2(0,1).val_.d_ = 1.0;
   mv_2(1,0).val_.d_ = 1.0;
   mv_2(1,1).val_.d_ = 1.0;
  
  matrix_ffv output_v = add(mv_1, md_2);
  output_v = add(mv_1, mv_2);

  AVEC q = createAVEC(mv_1(0,0).val().val(),mv_1(0,1).val().val(),mv_1(1,0).val().val(),mv_1(1,1).val().val());
  VEC h;
  output_v(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorAddition,ffv_matrix_matrix_exception) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  
  matrix_d d1(2,2), d2(1,2);
  matrix_ffv v1(2,2), v2(1,2);

  EXPECT_THROW(add(d1, d2), std::invalid_argument);
  EXPECT_THROW(add(d1, v2), std::invalid_argument);
  EXPECT_THROW(add(v1, d2), std::invalid_argument);
  EXPECT_THROW(add(v1, v2), std::invalid_argument);
}
