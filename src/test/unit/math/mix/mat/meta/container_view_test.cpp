#include <gtest/gtest.h>
#include <stan/math/prim/mat/meta/container_view.hpp>
#include <boost/type_traits/conditional.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>

TEST(MathMeta, container_view_vector_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  var* y = static_cast<var*>
                       (stan::math::chainable::operator new
                        (sizeof(var) * 10));
  Matrix<fvar<var>, Dynamic, 1> x(10);
  container_view<Matrix<fvar<var>, Dynamic, 1>, var> view_test(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test[i] = i;
    EXPECT_FLOAT_EQ(i, view_test[i].val());
    EXPECT_FLOAT_EQ(i, y[i].val());
    view_test[i] = 0;
    EXPECT_FLOAT_EQ(0, y[i].val());
  }
  
  container_view<Matrix<fvar<var>, Dynamic, 1>, Matrix<var, Dynamic, 1> > view_test_vec(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test_vec[i] << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    for (int j = 0; j < 10; ++j) {
      EXPECT_FLOAT_EQ(j, y[j].val());
      EXPECT_FLOAT_EQ(j, view_test_vec[i](j).val());
    }
  }
}

TEST(MathMeta, container_view_vector_fvar_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  fvar<var>* y = static_cast<fvar<var>*>
                       (stan::math::chainable::operator new
                        (sizeof(fvar<var>) * 10));
  Matrix<fvar<fvar<var> >, Dynamic, 1> x(10);
  container_view<Matrix<fvar<fvar<var> >, Dynamic, 1>, fvar<var> > view_test(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test[i] = i;
    EXPECT_FLOAT_EQ(i, view_test[i].val_.val());
    EXPECT_FLOAT_EQ(i, y[i].val_.val());
    view_test[i] = 0;
    EXPECT_FLOAT_EQ(0, y[i].val_.val());
  }
  
  container_view<Matrix<fvar<fvar<var> >, Dynamic, 1>, Matrix<fvar<var>, Dynamic, 1> > view_test_vec(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test_vec[i] << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    for (int j = 0; j < 10; ++j) {
      EXPECT_FLOAT_EQ(j, y[j].val_.val());
      EXPECT_FLOAT_EQ(j, view_test_vec[i](j).val_.val());
    }
  }
}

TEST(MathMeta, container_view_row_vector_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  var* y = static_cast<var*>
                       (stan::math::chainable::operator new
                        (sizeof(var) * 10));
  Matrix<fvar<var>, 1, Dynamic> x(10);
  container_view<Matrix<fvar<var>, 1, Dynamic>, var> view_test(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test[i] = i;
    EXPECT_FLOAT_EQ(i, view_test[i].val());
    EXPECT_FLOAT_EQ(i, y[i].val());
    view_test[i] = 0;
    EXPECT_FLOAT_EQ(0, y[i].val());
  }
  
  container_view<Matrix<fvar<var>, 1, Dynamic>, Matrix<var, 1, Dynamic> > view_test_vec(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test_vec[i] << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    for (int j = 0; j < 10; ++j) {
      EXPECT_FLOAT_EQ(j, y[j].val());
      EXPECT_FLOAT_EQ(j, view_test_vec[i](j).val());
    }
  }
}

TEST(MathMeta, container_view_row_vector_fvar_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  fvar<var>* y = static_cast<fvar<var>*>
                       (stan::math::chainable::operator new
                        (sizeof(fvar<var>) * 10));
  Matrix<fvar<fvar<var> >, 1, Dynamic> x(10);
  container_view<Matrix<fvar<fvar<var> >, 1, Dynamic>, fvar<var> > view_test(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test[i] = i;
    EXPECT_FLOAT_EQ(i, view_test[i].val_.val());
    EXPECT_FLOAT_EQ(i, y[i].val_.val());
    view_test[i] = 0;
    EXPECT_FLOAT_EQ(0, y[i].val_.val());
  }
  
  container_view<Matrix<fvar<fvar<var> >, 1, Dynamic>, Matrix<fvar<var>, 1, Dynamic> > view_test_vec(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test_vec[i] << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    for (int j = 0; j < 10; ++j) {
      EXPECT_FLOAT_EQ(j, y[j].val_.val());
      EXPECT_FLOAT_EQ(j, view_test_vec[i](j).val_.val());
    }
  }
}

TEST(MathMeta, container_view_matrix_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  var* y = static_cast<var*>
                       (stan::math::chainable::operator new
                        (sizeof(var) * 9));
  Matrix<fvar<var>, Dynamic, Dynamic> x(3,3);
  container_view<Matrix<fvar<var>, Dynamic, Dynamic>, Matrix<var, Dynamic, Dynamic> > view_test(x, y);
  int fill = 0;
  int matindex = 10;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      matindex *= j;
      view_test[i] << 0, 3, 6,
                      1, 4, 7,
                      2, 5, 8;
      EXPECT_FLOAT_EQ(fill++, view_test[matindex](i, j).val());
    }
  }
}

TEST(MathMeta, container_view_matrix_fvar_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  fvar<var>* y = static_cast<fvar<var>*>
                       (stan::math::chainable::operator new
                        (sizeof(fvar<var>) * 9));
  Matrix<fvar<fvar<var> >, Dynamic, Dynamic> x(3,3);
  container_view<Matrix<fvar<fvar<var> >, Dynamic, Dynamic>, Matrix<fvar<var>, Dynamic, Dynamic> > view_test(x, y);
  int fill = 0;
  int matindex = 10;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      matindex *= j;
      view_test[i] << 0, 3, 6,
                      1, 4, 7,
                      2, 5, 8;
      EXPECT_FLOAT_EQ(fill++, view_test[matindex](i, j).val_.val());
    }
  }
}

TEST(MathMeta, container_view_vector_vector_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  var* y = static_cast<var*>
                       (stan::math::chainable::operator new
                        (sizeof(var) * 15));
  std::vector<Matrix<fvar<var>, Dynamic, 1> > x;
  x.push_back(Matrix<fvar<var>, Dynamic, 1>(5));
  x.push_back(Matrix<fvar<var>, Dynamic, 1>(5));
  x.push_back(Matrix<fvar<var>, Dynamic, 1>(5));
  container_view<std::vector<Matrix<fvar<var>, Dynamic, 1> >, Matrix<var, Dynamic, 1> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    view_test[i] << 0, 1, 2, 3, 4;
    for (int j = 0; j < 5; ++j) {
      EXPECT_FLOAT_EQ(j, view_test[i](j).val());
      EXPECT_FLOAT_EQ(j, y[i * 5 + j].val());
    }
  }
}

TEST(MathMeta, container_view_vector_vector_fvar_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  fvar<var>* y = static_cast<fvar<var>*>
                       (stan::math::chainable::operator new
                        (sizeof(fvar<var>) * 15));
  std::vector<Matrix<fvar<fvar<var> >, Dynamic, 1> > x;
  x.push_back(Matrix<fvar<fvar<var> >, Dynamic, 1>(5));
  x.push_back(Matrix<fvar<fvar<var> >, Dynamic, 1>(5));
  x.push_back(Matrix<fvar<fvar<var> >, Dynamic, 1>(5));
  container_view<std::vector<Matrix<fvar<fvar<var> >, Dynamic, 1> >, Matrix<fvar<var>, Dynamic, 1> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    view_test[i] << 0, 1, 2, 3, 4;
    for (int j = 0; j < 5; ++j) {
      EXPECT_FLOAT_EQ(j, view_test[i](j).val_.val());
      EXPECT_FLOAT_EQ(j, y[i * 5 + j].val_.val());
    }
  }
}

TEST(MathMeta, container_view_vector_row_vector_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  var* y = static_cast<var*>
                       (stan::math::chainable::operator new
                        (sizeof(var) * 15));
  std::vector<Matrix<fvar<var>, 1, Dynamic> > x;
  x.push_back(Matrix<fvar<var>, 1, Dynamic>(5));
  x.push_back(Matrix<fvar<var>, 1, Dynamic>(5));
  x.push_back(Matrix<fvar<var>, 1, Dynamic>(5));
  container_view<std::vector<Matrix<fvar<var>, 1, Dynamic> >, Matrix<var, 1, Dynamic> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    view_test[i] << 0, 1, 2, 3, 4;
    for (int j = 0; j < 5; ++j) {
      EXPECT_FLOAT_EQ(j, view_test[i](j).val());
      EXPECT_FLOAT_EQ(j, y[i * 5 + j].val());
    }
  }
}

TEST(MathMeta, container_view_vector_row_vector_fvar_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  fvar<var>* y = static_cast<fvar<var>*>
                       (stan::math::chainable::operator new
                        (sizeof(fvar<var>) * 15));
  std::vector<Matrix<fvar<fvar<var> >, 1, Dynamic> > x;
  x.push_back(Matrix<fvar<fvar<var> >, 1, Dynamic>(5));
  x.push_back(Matrix<fvar<fvar<var> >, 1, Dynamic>(5));
  x.push_back(Matrix<fvar<fvar<var> >, 1, Dynamic>(5));
  container_view<std::vector<Matrix<fvar<fvar<var> >, 1, Dynamic> >, Matrix<fvar<var>, 1, Dynamic> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    view_test[i] << 0, 1, 2, 3, 4;
    for (int j = 0; j < 5; ++j) {
      EXPECT_FLOAT_EQ(j, view_test[i](j).val_.val());
      EXPECT_FLOAT_EQ(j, y[i * 5 + j].val_.val());
    }
  }
}

TEST(MathMeta, container_view_vector_matrix_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  var* y = static_cast<var*>
                       (stan::math::chainable::operator new
                        (sizeof(var) * 27));
  std::vector<Matrix<fvar<var>, Dynamic, Dynamic> > x;
  x.push_back(Matrix<fvar<var>, Dynamic, Dynamic>(3, 3));
  x.push_back(Matrix<fvar<var>, Dynamic, Dynamic>(3, 3));
  x.push_back(Matrix<fvar<var>, Dynamic, Dynamic>(3, 3));
  container_view<std::vector<Matrix<fvar<var>, Dynamic, Dynamic> >, Matrix<var, Dynamic, Dynamic> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    int j = -1;
    view_test[i] << 0, 3, 6,
                    1, 4, 7,
                    2, 5, 8;
    for (int m = 0; m < 3; ++m) {
      for (int n = 0; n < 3; ++n) {
        EXPECT_FLOAT_EQ(++j, view_test[i](n, m).val());
        EXPECT_FLOAT_EQ(j, y[i * 9 + m * 3 + n].val());
      }
    }
  }
}

TEST(MathMeta, container_view_vector_matrix_fvar_var) {
  using stan::math::container_view;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::fvar;

  fvar<var>* y = static_cast<fvar<var>*>
                       (stan::math::chainable::operator new
                        (sizeof(fvar<var>) * 27));
  std::vector<Matrix<fvar<fvar<var> >, Dynamic, Dynamic> > x;
  x.push_back(Matrix<fvar<fvar<var> >, Dynamic, Dynamic>(3, 3));
  x.push_back(Matrix<fvar<fvar<var> >, Dynamic, Dynamic>(3, 3));
  x.push_back(Matrix<fvar<fvar<var> >, Dynamic, Dynamic>(3, 3));
  container_view<std::vector<Matrix<fvar<fvar<var> >, Dynamic, Dynamic> >, Matrix<fvar<var>, Dynamic, Dynamic> > view_test(x, y);
  for (int i = 0; i < 3; ++i) {
    int j = -1;
    view_test[i] << 0, 3, 6,
                    1, 4, 7,
                    2, 5, 8;
    for (int m = 0; m < 3; ++m) {
      for (int n = 0; n < 3; ++n) {
        EXPECT_FLOAT_EQ(++j, view_test[i](n, m).val_.val());
        EXPECT_FLOAT_EQ(j, y[i * 9 + m * 3 + n].val_.val());
      }
    }
  }
}

TEST(MathMeta, container_view_no_throw_vector_matrix_var) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::container_view;
  using boost::conditional;
  using stan::math::dummy;
  using stan::math::var;
  using stan::math::fvar;
  using stan::is_constant_struct;

  var* arr = static_cast<var*>
             (stan::math::chainable::operator new
              (sizeof(var) * 1));
  std::vector<Matrix<fvar<var>, Dynamic, Dynamic> > x;
  x.push_back(Matrix<fvar<var>, Dynamic, Dynamic>(1,1));
  container_view<conditional<is_constant_struct<std::vector<Matrix<fvar<var>, Dynamic, Dynamic> > >::value,dummy,std::vector<Matrix<fvar<var>, Dynamic, Dynamic> > >::type, var> view_test(x, arr);
  EXPECT_NO_THROW(view_test[0]);
}

TEST(MathMeta, container_view_no_throw_vector_matrix_matrix_view) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::container_view;
  using boost::conditional;
  using stan::math::dummy;
  using stan::math::var;
  using stan::math::fvar;
  using stan::is_constant_struct;

  var* arr = static_cast<var*>
             (stan::math::chainable::operator new
              (sizeof(var) * 1));
  std::vector<Matrix<fvar<var>, Dynamic, Dynamic> > x;
  x.push_back(Matrix<fvar<var>, Dynamic, Dynamic>(1,1));
  container_view<conditional<is_constant_struct<std::vector<Matrix<fvar<var>, Dynamic, Dynamic> > >::value,dummy,std::vector<Matrix<fvar<var>, Dynamic, Dynamic> > >::type, Matrix<var, Dynamic, Dynamic> > view_test(x, arr);
  EXPECT_NO_THROW(view_test[0]);
}

TEST(MathMeta, container_view_no_throw_vector_matrix_fvar_var) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::container_view;
  using boost::conditional;
  using stan::math::dummy;
  using stan::math::var;
  using stan::math::fvar;
  using stan::is_constant_struct;

  fvar<var>* arr = static_cast<fvar<var>*>
                       (stan::math::chainable::operator new
                        (sizeof(fvar<var>) * 1));
  std::vector<Matrix<fvar<fvar<var> >, Dynamic, Dynamic> > x;
  x.push_back(Matrix<fvar<fvar<var> >, Dynamic, Dynamic>(1,1));
  container_view<conditional<is_constant_struct<std::vector<Matrix<fvar<fvar<var> >, Dynamic, Dynamic> > >::value,dummy,std::vector<Matrix<fvar<fvar<var> >, Dynamic, Dynamic> > >::type, fvar<var> > view_test(x, arr);
  EXPECT_NO_THROW(view_test[0]);
}
