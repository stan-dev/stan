#include <stan/math/matrix/sd.hpp>
#include <gtest/gtest.h>
#include <test/diff/util.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/diff/rev/matrix/typedefs.hpp>
#include <stan/diff.hpp>

TEST(DiffRevMatrix, sd_vector) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::diff::vector_v;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  vector_d d1(6);
  vector_v v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val());
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val());
}
TEST(DiffRevMatrix, sd_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::diff::vector_v;

  vector_d d1;
  vector_v v1;
  EXPECT_THROW(sd(d1), std::domain_error);
  EXPECT_THROW(sd(v1), std::domain_error);
}
TEST(DiffRevMatrix, sd_rowvector) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::diff::row_vector_v;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_v v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val());

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val());
}
TEST(DiffRevMatrix, sd_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::diff::row_vector_v;

  row_vector_d d;
  row_vector_v v;
  
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(DiffRevMatrix, sd_matrix) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::diff::matrix_v;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_v v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val());

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val());
}
TEST(DiffRevMatrix, sd_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::diff::matrix_v;

  matrix_d d;
  matrix_v v;

  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(DiffRevMatrix, sdStdVector) {
  using stan::math::sd; // should use arg-dep lookup (and for sqrt)

  AVEC y1 = createAVEC(0.5,2.0,3.5);
  AVAR f1 = sd(y1);
  VEC grad1 = cgrad(f1, y1[0], y1[1], y1[2]);
  double f1_val = f1.val(); // save before cleaned out

  AVEC y2 = createAVEC(0.5,2.0,3.5);
  AVAR mean2 = (y2[0] + y2[1] + y2[2]) / 3.0;
  AVAR sum_sq_diff_2 
    = (y2[0] - mean2) * (y2[0] - mean2)
    + (y2[1] - mean2) * (y2[1] - mean2)
    + (y2[2] - mean2) * (y2[2] - mean2); 
  AVAR f2 = sqrt(sum_sq_diff_2 / (3 - 1));

  EXPECT_EQ(f2.val(), f1_val);

  VEC grad2 = cgrad(f2, y2[0], y2[1], y2[2]);

  EXPECT_EQ(3U, grad1.size());
  EXPECT_EQ(3U, grad2.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(grad2[i], grad1[i]);
}
