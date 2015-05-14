#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/add.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradRevMatrix,add_scalar) {
  using stan::math::matrix_v;
  using stan::math::add;

  matrix_v v(2,2);
  v << 1, 2, 3, 4;
  matrix_v result;

  result = add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val());

  result = add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val());
}

TEST(AgradRevMatrix, add_vector_vector) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_d vd_1(5);
  vector_d vd_2(5);
  vector_v vv_1(5);
  vector_v vv_2(5);
  
  vd_1 << 1, 2, 3, 4, 5;
  vv_1 << 1, 2, 3, 4, 5;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
  
  vector_d expected_output(5);
  expected_output << 3, 5, 7, 9, 11;
  
  vector_d output_d;
  output_d = add(vd_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  vector_v output_v = add(vv_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  

  output_v = add(vd_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  

  output_v = add(vv_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  
}
TEST(AgradRevMatrix, add_vector_vector_exception) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_d d1(5), d2(1);
  vector_v v1(5), v2(1);
  
  EXPECT_THROW(add(d1, d2), std::invalid_argument);
  EXPECT_THROW(add(v1, d2), std::invalid_argument);
  EXPECT_THROW(add(d1, v2), std::invalid_argument);
  EXPECT_THROW(add(v1, v2), std::invalid_argument);
}
TEST(AgradRevMatrix, add_rowvector_rowvector) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;

  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_v rvv_1(5), rvv_2(5);

  rvd_1 << 1, 2, 3, 4, 5;
  rvv_1 << 1, 2, 3, 4, 5;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
  
  row_vector_d expected_output(5);
  expected_output << 3, 5, 7, 9, 11;
  
  row_vector_d output_d = add(rvd_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  row_vector_v output_v = add(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  

  output_v = add(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  

  output_v = add(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  
}
TEST(AgradRevMatrix, add_rowvector_rowvector_exception) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;

  row_vector_d d1(5), d2(2);
  row_vector_v v1(5), v2(2);

  row_vector_v output;
  EXPECT_THROW( add(d1, d2), std::invalid_argument);
  EXPECT_THROW( add(d1, v2), std::invalid_argument);
  EXPECT_THROW( add(v1, d2), std::invalid_argument);
  EXPECT_THROW( add(v1, v2), std::invalid_argument);
}
TEST(AgradRevMatrix, add_matrix_matrix) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_v;

  matrix_d md_1(2,2), md_2(2,2);
  matrix_v mv_1(2,2), mv_2(2,2);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
  
  matrix_d expected_output(2,2);
  expected_output << 0, -9, 11, 2;
  
  matrix_d output_d = add(md_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_d(0,0));
  EXPECT_FLOAT_EQ(expected_output(0,1), output_d(0,1));
  EXPECT_FLOAT_EQ(expected_output(1,0), output_d(1,0));
  EXPECT_FLOAT_EQ(expected_output(1,1), output_d(1,1));

  matrix_v output_v = add(mv_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val());

  output_v = add(md_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val());

  output_v = add(mv_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val());
}
TEST(AgradRevMatrix, add_matrix_matrix_exception) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  
  matrix_d d1(2,2), d2(1,2);
  matrix_v v1(2,2), v2(1,2);

  EXPECT_THROW(add(d1, d2), std::invalid_argument);
  EXPECT_THROW(add(d1, v2), std::invalid_argument);
  EXPECT_THROW(add(v1, d2), std::invalid_argument);
  EXPECT_THROW(add(v1, v2), std::invalid_argument);
}
