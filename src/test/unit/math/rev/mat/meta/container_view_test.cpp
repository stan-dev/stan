#include <gtest/gtest.h>
#include <stan/math/prim/mat/meta/container_view.hpp>
#include <boost/type_traits/conditional.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/rev/core.hpp>

TEST(MathMeta, container_view_vector) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  double y[10];
  Matrix<var, Dynamic, 1> x(10);
  container_view<Matrix<var, Dynamic, 1>, double> view_test(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test[i] = i;
    EXPECT_FLOAT_EQ(i, view_test[i]);
    EXPECT_FLOAT_EQ(i, y[i]);
    view_test[i] = 0;
    EXPECT_FLOAT_EQ(0, y[i]);
  }
  
  container_view<Matrix<var, Dynamic, 1>, Matrix<double, Dynamic, 1> > view_test_vec(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test_vec[i] << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    for (int j = 0; j < 10; ++j) {
      EXPECT_FLOAT_EQ(j, y[j]);
      EXPECT_FLOAT_EQ(j, view_test_vec[i](j));
    }
  }
}

TEST(MathMeta, container_view_row_vector) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  double y[10];
  Matrix<var, 1, Dynamic> x(10);
  container_view<Matrix<var, 1, Dynamic>, double> view_test(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test[i] = i;
    EXPECT_FLOAT_EQ(i, view_test[i]);
    EXPECT_FLOAT_EQ(i, y[i]);
    view_test[i] = 0;
    EXPECT_FLOAT_EQ(0, y[i]);
  }
  
  container_view<Matrix<var, 1, Dynamic>, Matrix<double, 1, Dynamic> > view_test_vec(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test_vec[i] << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    for (int j = 0; j < 10; ++j) {
      EXPECT_FLOAT_EQ(j, y[j]);
      EXPECT_FLOAT_EQ(j, view_test_vec[i](j));
    }
  }
}

TEST(MathMeta, container_view_matrix) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  double y[9];
  Matrix<var, Dynamic, Dynamic> x(3,3);
  container_view<Matrix<var, Dynamic, Dynamic>, Matrix<double, Dynamic, Dynamic> > view_test(x, y);
  int fill = 0;
  int matindex = 10;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      matindex *= j;
      view_test[i] << 0, 3, 6,
                      1, 4, 7,
                      2, 5, 8;
      EXPECT_FLOAT_EQ(fill++, view_test[matindex](i, j));
    }
  }
}

TEST(MathMeta, container_view_vector_vector) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  double y[15];
  std::vector<Matrix<var, Dynamic, 1> > x;
  x.push_back(Matrix<var, Dynamic, 1>(5));
  x.push_back(Matrix<var, Dynamic, 1>(5));
  x.push_back(Matrix<var, Dynamic, 1>(5));
  container_view<std::vector<Matrix<var, Dynamic, 1> >, Matrix<double, Dynamic, 1> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    view_test[i] << 0, 1, 2, 3, 4;
    for (int j = 0; j < 5; ++j) {
      EXPECT_FLOAT_EQ(j, view_test[i](j));
      EXPECT_FLOAT_EQ(j, y[i * 5 + j]);
    }
  }
}

TEST(MathMeta, container_view_vector_row_vector) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  double y[15];
  std::vector<Matrix<var, 1, Dynamic> > x;
  x.push_back(Matrix<var, 1, Dynamic>(5));
  x.push_back(Matrix<var, 1, Dynamic>(5));
  x.push_back(Matrix<var, 1, Dynamic>(5));
  container_view<std::vector<Matrix<var, 1, Dynamic> >, Matrix<double, 1, Dynamic> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    view_test[i] << 0, 1, 2, 3, 4;
    for (int j = 0; j < 5; ++j) {
      EXPECT_FLOAT_EQ(j, view_test[i](j));
      EXPECT_FLOAT_EQ(j, y[i * 5 + j]);
    }
  }
}

TEST(MathMeta, container_view_vector_matrix) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  double y[27];
  std::vector<Matrix<var, Dynamic, Dynamic> > x;
  x.push_back(Matrix<var, Dynamic, Dynamic>(3, 3));
  x.push_back(Matrix<var, Dynamic, Dynamic>(3, 3));
  x.push_back(Matrix<var, Dynamic, Dynamic>(3, 3));
  container_view<std::vector<Matrix<var, Dynamic, Dynamic> >, Matrix<double, Dynamic, Dynamic> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    int j = -1;
    view_test[i] << 0, 3, 6,
                    1, 4, 7,
                    2, 5, 8;
    for (int m = 0; m < 3; ++m) {
      for (int n = 0; n < 3; ++n) {
        EXPECT_FLOAT_EQ(++j, view_test[i](n, m));
        EXPECT_FLOAT_EQ(j, y[i * 9 + m * 3 + n]);
      }
    }
  }
}

TEST(MathMeta, container_view_no_throw_vector_matrix) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::container_view;
  using boost::conditional;
  using stan::math::dummy;
  using stan::math::var;
  using stan::is_constant_struct;

  double arr[1];
  std::vector<Matrix<var, Dynamic, Dynamic> > x;
  x.push_back(Matrix<var, Dynamic, Dynamic>(1,1));
  container_view<conditional<is_constant_struct<std::vector<Matrix<var, Dynamic, Dynamic> > >::value,dummy,std::vector<Matrix<var, Dynamic, Dynamic> > >::type, double> view_test(x, arr);
  EXPECT_NO_THROW(view_test[0]);
}
