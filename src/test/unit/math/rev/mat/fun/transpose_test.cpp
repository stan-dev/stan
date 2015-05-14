#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

TEST(AgradRevMatrix,transpose_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  using stan::math::transpose;

  EXPECT_EQ(0,transpose(matrix_v()).size());
  EXPECT_EQ(0,transpose(matrix_d()).size());

  matrix_v a(2,3);
  a << -1.0, 2.0, -3.0, 
    5.0, 10.0, 100.0;
  
  AVEC x = createAVEC(a(0,0), a(0,2), a(1,1));
  
  matrix_v c = transpose(a);
  EXPECT_FLOAT_EQ(-1.0,c(0,0).val());
  EXPECT_FLOAT_EQ(10.0,c(1,1).val());
  EXPECT_FLOAT_EQ(-3.0,c(2,0).val());
  EXPECT_EQ(3,c.rows());
  EXPECT_EQ(2,c.cols());

  VEC g = cgradvec(c(2,0),x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}
TEST(AgradRevMatrix,transpose_vector) {
  using stan::math::vector_v;
  using stan::math::row_vector_v;
  using stan::math::transpose;

  vector_v a(3);
  a << 1.0, 2.0, 3.0;
  
  AVEC x = createAVEC(a(0),a(1),a(2));

  row_vector_v a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val(),a_tr(i).val());

  VEC g = cgradvec(a_tr(1),x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}
TEST(AgradRevMatrix,transpose_row_vector) {
  using stan::math::vector_v;
  using stan::math::row_vector_v;
  using stan::math::transpose;

  row_vector_v a(3);
  a << 1.0, 2.0, 3.0;
  
  AVEC x = createAVEC(a(0),a(1),a(2));

  vector_v a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_type i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val(),a_tr(i).val());

  VEC g = cgradvec(a_tr(1),x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}
