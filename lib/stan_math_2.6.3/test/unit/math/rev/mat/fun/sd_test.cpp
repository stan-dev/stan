#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <gtest/gtest.h>

#include <iostream>

#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/mat/fun/sd.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/variance.hpp>
#include <stan/math/prim/mat/fun/sd.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>

TEST(AgradRevMatrix, sd_eq) {
  using stan::math::sd;
  using stan::math::var;
  using stan::math::vector_v;
  using std::vector;

  for (size_t size = 2; size <= 200; size *= 3) {
    vector<var> x_std_vec(size);
    vector_v x(size);
    for (size_t i = 0; i < size; ++i) {
      x(i) = 3;
      x_std_vec[i] = x(i);
    }

    stan::math::var f = sd(x);
    EXPECT_NEAR(0.0, f.val(), 1e-12);

    vector<double> grad;
    f.grad(x_std_vec,grad);

    EXPECT_EQ(size, grad.size());
    double analytic = std::sqrt(size) / size;
    for (size_t j = 0; j < size; ++j)
      EXPECT_FLOAT_EQ(analytic, grad[j]);
  }
}

TEST(AgradRevMatrix, sd_vector) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::math::vector_v;

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
TEST(AgradRevMatrix, sd_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_d d1;
  vector_v v1;
  EXPECT_THROW(sd(d1), std::invalid_argument);
  EXPECT_THROW(sd(v1), std::invalid_argument);
}
TEST(AgradRevMatrix, sd_rowvector) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;

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
TEST(AgradRevMatrix, sd_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;

  row_vector_d d;
  row_vector_v v;
  
  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);
}
TEST(AgradRevMatrix, sd_matrix) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::math::matrix_v;

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
TEST(AgradRevMatrix, sd_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::math::matrix_v;

  matrix_d d;
  matrix_v v;

  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);
}
TEST(AgradRevMatrix, sdStdVector) {
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
// used to validate analytic gradient definition at limit sd(x) -> 0


TEST(AgradRevSd, finiteDiffsMatchAnalytic) {
  using std::sqrt;
  using stan::math::sd;
  for (int n = 2; n <= 128; n *= 2) {
    double analytic = sqrt(n) / n;
    double epsilon = 1e-7;
    std::vector<double> y(n,1.0);
    double sd_y = 0.0;
    y[1] += epsilon;
    double sd_y_plus_epsilon = sd(y);
    double finite_diff = (sd_y_plus_epsilon - sd_y) / epsilon;
    EXPECT_FLOAT_EQ(analytic, finite_diff);
  }
}
