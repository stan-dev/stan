#include <gtest/gtest.h>
#include <stan/math/prim/mat/meta/container_view.hpp>
#include <boost/type_traits/conditional.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>

TEST(MathMeta, container_view_vector) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[10];
  Matrix<double, Dynamic, 1> x(10);
  container_view<Matrix<double, Dynamic, 1>, double> view_test(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test[i] = i;
    EXPECT_FLOAT_EQ(i, view_test[i]);
    EXPECT_FLOAT_EQ(i, y[i]);
    view_test[i] = 0;
    EXPECT_FLOAT_EQ(0, y[i]);
  }
  
  container_view<Matrix<double, Dynamic, 1>, Matrix<double, Dynamic, 1> > view_test_vec(x, y);
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

  double y[10];
  Matrix<double, 1, Dynamic> x(10);
  container_view<Matrix<double, 1, Dynamic>, double> view_test(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test[i] = i;
    EXPECT_FLOAT_EQ(i, view_test[i]);
    EXPECT_FLOAT_EQ(i, y[i]);
    view_test[i] = 0;
    EXPECT_FLOAT_EQ(0, y[i]);
  }
  
  container_view<Matrix<double, 1, Dynamic>, Matrix<double, 1, Dynamic> > view_test_vec(x, y);
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

  double y[9];
  Matrix<double, Dynamic, Dynamic> x(3,3);
  container_view<Matrix<double, Dynamic, Dynamic>, Matrix<double, Dynamic, Dynamic> > view_test(x, y);
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

TEST(MathMeta, container_view_throw) {
  using stan::math::container_view;
  using boost::conditional;
  using stan::math::dummy;
  using stan::is_constant_struct;

  double arr[1];
  container_view<conditional<is_constant_struct<double>::value,dummy,double>::type, double> view_test(4.0, arr);
  EXPECT_THROW(view_test[0],std::out_of_range);
}
