#include <stan/math/matrix/dims.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>
#include <gtest/gtest.h>

using stan::agrad::fvar;
using stan::agrad::var;

TEST(AgradFwdMatrixDims, matrix_fd) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::dims;
  using stan::agrad::matrix_fd;

  fvar<double> x1;
  vector<int> dims1 = dims(x1);
  EXPECT_EQ(0U,dims1.size());

  vector<fvar<double> > x3;
  x3.push_back(-32.1); x3.push_back(17.9);
  vector<int> dims3 = dims(x3);
  EXPECT_EQ(1U,dims3.size());
  EXPECT_EQ(2,dims3[0]);

  vector<vector<fvar<double> > > x4;
  x4.push_back(x3);  x4.push_back(x3);   x4.push_back(x3);
  vector<int> dims4 = dims(x4);
  EXPECT_EQ(2U,dims4.size());
  EXPECT_EQ(3,dims4[0]);
  EXPECT_EQ(2,dims4[1]);

  Matrix<fvar<double>,Dynamic,Dynamic> x5(7,8);
  vector<int> dims5 = dims(x5);
  EXPECT_EQ(2U,dims5.size());
  EXPECT_EQ(7,dims5[0]);
  EXPECT_EQ(8,dims5[1]);

  Matrix<fvar<double>,Dynamic,1> x6(17);
  vector<int> dims6 = dims(x6);
  EXPECT_EQ(2U,dims6.size());
  EXPECT_EQ(17,dims6[0]);
  EXPECT_EQ(1,dims6[1]);

  Matrix<fvar<double>,1,Dynamic> x7(17);
  vector<int> dims7 = dims(x7);
  EXPECT_EQ(2U,dims7.size());
  EXPECT_EQ(1,dims7[0]);
  EXPECT_EQ(17,dims7[1]);

  vector<Matrix<fvar<double>,Dynamic,Dynamic> > x8;
  x8.push_back(x5);  x8.push_back(x5);
  vector<int> dims8 = dims(x8);
  EXPECT_EQ(3U,dims8.size());
  EXPECT_EQ(2,dims8[0]);
  EXPECT_EQ(7,dims8[1]);
  EXPECT_EQ(8,dims8[2]);
}

TEST(AgradFwdMatrixDims, matrix_fv) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::dims;
  using stan::agrad::matrix_fv;

  fvar<var> x1;
  vector<int> dims1 = dims(x1);
  EXPECT_EQ(0U,dims1.size());

  vector<fvar<var> > x3;
  x3.push_back(-32.1); x3.push_back(17.9);
  vector<int> dims3 = dims(x3);
  EXPECT_EQ(1U,dims3.size());
  EXPECT_EQ(2,dims3[0]);

  vector<vector<fvar<var> > > x4;
  x4.push_back(x3);  x4.push_back(x3);   x4.push_back(x3);
  vector<int> dims4 = dims(x4);
  EXPECT_EQ(2U,dims4.size());
  EXPECT_EQ(3,dims4[0]);
  EXPECT_EQ(2,dims4[1]);

  Matrix<fvar<var>,Dynamic,Dynamic> x5(7,8);
  vector<int> dims5 = dims(x5);
  EXPECT_EQ(2U,dims5.size());
  EXPECT_EQ(7,dims5[0]);
  EXPECT_EQ(8,dims5[1]);

  Matrix<fvar<var>,Dynamic,1> x6(17);
  vector<int> dims6 = dims(x6);
  EXPECT_EQ(2U,dims6.size());
  EXPECT_EQ(17,dims6[0]);
  EXPECT_EQ(1,dims6[1]);

  Matrix<fvar<var>,1,Dynamic> x7(17);
  vector<int> dims7 = dims(x7);
  EXPECT_EQ(2U,dims7.size());
  EXPECT_EQ(1,dims7[0]);
  EXPECT_EQ(17,dims7[1]);

  vector<Matrix<fvar<var>,Dynamic,Dynamic> > x8;
  x8.push_back(x5);  x8.push_back(x5);
  vector<int> dims8 = dims(x8);
  EXPECT_EQ(3U,dims8.size());
  EXPECT_EQ(2,dims8[0]);
  EXPECT_EQ(7,dims8[1]);
  EXPECT_EQ(8,dims8[2]);
}

TEST(AgradFwdMatrixDims, matrix_ffd) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::dims;
  using stan::agrad::matrix_ffd;

  fvar<fvar<double> > x1;
  vector<int> dims1 = dims(x1);
  EXPECT_EQ(0U,dims1.size());

  vector<fvar<fvar<double> > > x3;
  x3.push_back(-32.1); x3.push_back(17.9);
  vector<int> dims3 = dims(x3);
  EXPECT_EQ(1U,dims3.size());
  EXPECT_EQ(2,dims3[0]);

  vector<vector<fvar<fvar<double> > > > x4;
  x4.push_back(x3);  x4.push_back(x3);   x4.push_back(x3);
  vector<int> dims4 = dims(x4);
  EXPECT_EQ(2U,dims4.size());
  EXPECT_EQ(3,dims4[0]);
  EXPECT_EQ(2,dims4[1]);

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> x5(7,8);
  vector<int> dims5 = dims(x5);
  EXPECT_EQ(2U,dims5.size());
  EXPECT_EQ(7,dims5[0]);
  EXPECT_EQ(8,dims5[1]);

  Matrix<fvar<fvar<double> >,Dynamic,1> x6(17);
  vector<int> dims6 = dims(x6);
  EXPECT_EQ(2U,dims6.size());
  EXPECT_EQ(17,dims6[0]);
  EXPECT_EQ(1,dims6[1]);

  Matrix<fvar<fvar<double> >,1,Dynamic> x7(17);
  vector<int> dims7 = dims(x7);
  EXPECT_EQ(2U,dims7.size());
  EXPECT_EQ(1,dims7[0]);
  EXPECT_EQ(17,dims7[1]);

  vector<Matrix<fvar<fvar<double> >,Dynamic,Dynamic> > x8;
  x8.push_back(x5);  x8.push_back(x5);
  vector<int> dims8 = dims(x8);
  EXPECT_EQ(3U,dims8.size());
  EXPECT_EQ(2,dims8[0]);
  EXPECT_EQ(7,dims8[1]);
  EXPECT_EQ(8,dims8[2]);
}

TEST(AgradFwdMatrixDims, matrix_ffv) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::dims;
  using stan::agrad::matrix_ffv;

  fvar<fvar<var> > x1;
  vector<int> dims1 = dims(x1);
  EXPECT_EQ(0U,dims1.size());

  vector<fvar<fvar<var> > > x3;
  x3.push_back(-32.1); x3.push_back(17.9);
  vector<int> dims3 = dims(x3);
  EXPECT_EQ(1U,dims3.size());
  EXPECT_EQ(2,dims3[0]);

  vector<vector<fvar<fvar<var> > > > x4;
  x4.push_back(x3);  x4.push_back(x3);   x4.push_back(x3);
  vector<int> dims4 = dims(x4);
  EXPECT_EQ(2U,dims4.size());
  EXPECT_EQ(3,dims4[0]);
  EXPECT_EQ(2,dims4[1]);

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> x5(7,8);
  vector<int> dims5 = dims(x5);
  EXPECT_EQ(2U,dims5.size());
  EXPECT_EQ(7,dims5[0]);
  EXPECT_EQ(8,dims5[1]);

  Matrix<fvar<fvar<var> >,Dynamic,1> x6(17);
  vector<int> dims6 = dims(x6);
  EXPECT_EQ(2U,dims6.size());
  EXPECT_EQ(17,dims6[0]);
  EXPECT_EQ(1,dims6[1]);

  Matrix<fvar<fvar<var> >,1,Dynamic> x7(17);
  vector<int> dims7 = dims(x7);
  EXPECT_EQ(2U,dims7.size());
  EXPECT_EQ(1,dims7[0]);
  EXPECT_EQ(17,dims7[1]);

  vector<Matrix<fvar<fvar<var> >,Dynamic,Dynamic> > x8;
  x8.push_back(x5);  x8.push_back(x5);
  vector<int> dims8 = dims(x8);
  EXPECT_EQ(3U,dims8.size());
  EXPECT_EQ(2,dims8[0]);
  EXPECT_EQ(7,dims8[1]);
  EXPECT_EQ(8,dims8[2]);
}
