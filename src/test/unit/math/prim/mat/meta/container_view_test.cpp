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

TEST(MathMeta, container_view_vector_zero_size) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[0];
  Matrix<double, Dynamic, 1> x(1);
  x.resize(0);
  container_view<Matrix<double, Dynamic, 1>, Matrix<double, Dynamic, 1> > view_test_vec(x, y);
  EXPECT_DEATH(view_test_vec[0](0), "");
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

TEST(MathMeta, container_view_row_vector_zero_size) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[0];
  Matrix<double, 1, Dynamic> x(1);
  x.resize(0);
  container_view<Matrix<double, 1, Dynamic>, Matrix<double, 1, Dynamic> > view_test_vec(x, y);
  EXPECT_DEATH(view_test_vec[0](0), "");
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

TEST(MathMeta, container_view_matrix_zero_size) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[0];
  Matrix<double, Dynamic, Dynamic> x(1,1);
  x.resize(0,0);
  container_view<Matrix<double, Dynamic, Dynamic>, Matrix<double, Dynamic, Dynamic> > view_test_vec(x, y);
  EXPECT_DEATH(view_test_vec[0](0,0), "");
}

TEST(MathMeta, container_view_vector_vector) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[15];
  std::vector<Matrix<double, Dynamic, 1> > x;
  x.push_back(Matrix<double, Dynamic, 1>(5));
  x.push_back(Matrix<double, Dynamic, 1>(5));
  x.push_back(Matrix<double, Dynamic, 1>(5));
  container_view<std::vector<Matrix<double, Dynamic, 1> >, Matrix<double, Dynamic, 1> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    view_test[i] << 0, 1, 2, 3, 4;
    for (int j = 0; j < 5; ++j) {
      EXPECT_FLOAT_EQ(j, view_test[i](j));
      EXPECT_FLOAT_EQ(j, y[i * 5 + j]);
    }
  }
}

TEST(MathMeta, container_view_vector_vector_zero_size) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[0];
  std::vector<Matrix<double, Dynamic, 1> > x;
  x.push_back(Matrix<double, Dynamic, 1>(5));
  x.push_back(Matrix<double, Dynamic, 1>(5));
  x.push_back(Matrix<double, Dynamic, 1>(5));
  x[0].resize(0);
  x[1].resize(0);
  x[2].resize(0);
  container_view<std::vector<Matrix<double, Dynamic, 1> >, Matrix<double, Dynamic, 1> > view_test(x, y);
  for (int i = 0; i < 3; ++i) 
    EXPECT_DEATH(view_test[i](0),"");
}

TEST(MathMeta, container_view_vector_row_vector) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[15];
  std::vector<Matrix<double, 1, Dynamic> > x;
  x.push_back(Matrix<double, 1, Dynamic>(5));
  x.push_back(Matrix<double, 1, Dynamic>(5));
  x.push_back(Matrix<double, 1, Dynamic>(5));
  container_view<std::vector<Matrix<double, 1, Dynamic> >, Matrix<double, 1, Dynamic> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    view_test[i] << 0, 1, 2, 3, 4;
    for (int j = 0; j < 5; ++j) {
      EXPECT_FLOAT_EQ(j, view_test[i](j));
      EXPECT_FLOAT_EQ(j, y[i * 5 + j]);
    }
  }
}

TEST(MathMeta, container_view_vector_row_vector_zero_size) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[0];
  std::vector<Matrix<double, 1, Dynamic> > x;
  x.push_back(Matrix<double, 1, Dynamic>(5));
  x.push_back(Matrix<double, 1, Dynamic>(5));
  x.push_back(Matrix<double, 1, Dynamic>(5));
  x[0].resize(0);
  x[1].resize(0);
  x[2].resize(0);
  container_view<std::vector<Matrix<double, 1, Dynamic> >, Matrix<double, 1, Dynamic> > view_test(x, y);
  for (int i = 0; i < 3; ++i) 
    EXPECT_DEATH(view_test[i](0),"");
}

TEST(MathMeta, container_view_vector_matrix) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[27];
  std::vector<Matrix<double, Dynamic, Dynamic> > x;
  x.push_back(Matrix<double, Dynamic, Dynamic>(3, 3));
  x.push_back(Matrix<double, Dynamic, Dynamic>(3, 3));
  x.push_back(Matrix<double, Dynamic, Dynamic>(3, 3));
  container_view<std::vector<Matrix<double, Dynamic, Dynamic> >, Matrix<double, Dynamic, Dynamic> > view_test(x, y);
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

TEST(MathMeta, container_view_vector_matrix_zero_size) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[0];
  std::vector<Matrix<double, Dynamic, Dynamic> > x;
  x.push_back(Matrix<double, Dynamic, Dynamic>(5,5));
  x.push_back(Matrix<double, Dynamic, Dynamic>(5,5));
  x.push_back(Matrix<double, Dynamic, Dynamic>(5,5));
  x[0].resize(0,0);
  x[1].resize(0,0);
  x[2].resize(0,0);
  container_view<std::vector<Matrix<double, Dynamic, Dynamic> >, Matrix<double, Dynamic, Dynamic> > view_test(x, y);
  for (int i = 0; i < 3; ++i) 
    EXPECT_DEATH(view_test[i](0,0),"");
}

TEST(MathMeta, container_view_zero_size_vector_matrix) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double y[0];
  std::vector<Matrix<double, Dynamic, Dynamic> > x;
  container_view<std::vector<Matrix<double, Dynamic, Dynamic> >, Matrix<double, Dynamic, Dynamic> > view_test(x, y);
  EXPECT_DEATH(view_test[0](0,0),"");
}

TEST(MathMeta, container_view_throw_matrix) {
  using stan::math::container_view;
  using boost::conditional;
  using stan::math::dummy;
  using stan::is_constant_struct;
  using Eigen::Dynamic;
  using Eigen::Matrix;

  double arr[1];
  container_view<conditional<is_constant_struct<Matrix<double, Dynamic, Dynamic> >::value,dummy,Matrix<double, Dynamic, Dynamic> >::type, double> view_test(4.0, arr);
  EXPECT_THROW(view_test[0],std::out_of_range);
}

TEST(MathMeta, container_view_throw_vector_matrix) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::container_view;
  using boost::conditional;
  using stan::math::dummy;
  using stan::is_constant_struct;

  double arr[1];
  container_view<conditional<is_constant_struct<std::vector<Matrix<double, Dynamic, Dynamic> > >::value,dummy,std::vector<Matrix<double, Dynamic, Dynamic> > >::type, double> view_test(4.0, arr);
  EXPECT_THROW(view_test[0],std::out_of_range);
}
