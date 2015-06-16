#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/elt_multiply.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradRevMatrix,elt_multiply_vec_vv) {
  using stan::math::elt_multiply;
  using stan::math::vector_v;

  vector_v x(2);
  x << 2, 5;
  vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1),y(0),y(1));
  vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(2.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}

TEST(AgradRevMatrix,elt_multiply_vec_vd) {
  using stan::math::elt_multiply;
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_v x(2);
  x << 2, 5;
  vector_d y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1));
  vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}
TEST(AgradRevMatrix,elt_multiply_vec_dv) {
  using stan::math::elt_multiply;
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_d x(2);
  x << 2, 5;
  vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(y(0),y(1));
  vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}

TEST(AgradRevMatrix,elt_multiply_row_vec_vv) {
  using stan::math::elt_multiply;
  using stan::math::row_vector_v;

  row_vector_v x(2);
  x << 2, 5;
  row_vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1),y(0),y(1));
  row_vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(2.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}
TEST(AgradRevMatrix,elt_multiply_row_vec_vd) {
  using stan::math::elt_multiply;
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;

  row_vector_v x(2);
  x << 2, 5;
  row_vector_d y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1));
  row_vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}
TEST(AgradRevMatrix,elt_multiply_row_vec_dv) {
  using stan::math::elt_multiply;
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;

  row_vector_d x(2);
  x << 2, 5;
  row_vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(y(0),y(1));
  row_vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}


TEST(AgradRevMatrix,elt_multiply_matrix_vv) {
  using stan::math::elt_multiply;
  using stan::math::matrix_v;

  matrix_v x(2,3);
  x << 2, 5, 6, 9, 13, 29;
  matrix_v y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
  AVEC x_ind = createAVEC(x(0,0),x(0,1),x(0,2),y(0,0));
  matrix_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0,0).val());
  EXPECT_FLOAT_EQ(500.0,z(0,1).val());
  EXPECT_FLOAT_EQ(29000000.0,z(1,2).val());

  VEC g = cgradvec(z(0,0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
  EXPECT_FLOAT_EQ(2.0,g[3]);
}
TEST(AgradRevMatrix,elt_multiply_matrix_vd) {
  using stan::math::elt_multiply;
  using stan::math::matrix_d;
  using stan::math::matrix_v;

  matrix_v x(2,3);
  x << 2, 5, 6, 9, 13, 29;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
  AVEC x_ind = createAVEC(x(0,0),x(0,1),x(0,2),x(1,0));
  matrix_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0,0).val());
  EXPECT_FLOAT_EQ(500.0,z(0,1).val());
  EXPECT_FLOAT_EQ(29000000.0,z(1,2).val());

  VEC g = cgradvec(z(0,0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}
TEST(AgradRevMatrix,elt_multiply_matrix_dv) {
  using stan::math::elt_multiply;
  using stan::math::matrix_d;
  using stan::math::matrix_v;

  matrix_d x(2,3);
  x << 2, 5, 6, 9, 13, 29;
  matrix_v y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
  AVEC x_ind = createAVEC(y(0,0),y(0,1));
  matrix_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0,0).val());
  EXPECT_FLOAT_EQ(500.0,z(0,1).val());
  EXPECT_FLOAT_EQ(29000000.0,z(1,2).val());

  VEC g = cgradvec(z(0,0),x_ind);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}
