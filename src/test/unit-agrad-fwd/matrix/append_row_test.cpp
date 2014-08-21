#include <stan/math/matrix/append_row.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>

TEST(AgradFwdMatrixAppendRow,fd) {
  using stan::math::append_row;
  using stan::agrad::matrix_fd;
  using Eigen::MatrixXd;

  matrix_fd a(2,2);
  MatrixXd ad(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  ad << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  a(0,0).d_ = 2.0;
  a(0,1).d_ = 3.0;
  a(1,0).d_ = 4.0;
  a(1,1).d_ = 5.0;

  matrix_fd ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_, ab_append_row(i ,j).d_);
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_, adb_append_row(i ,j));
}

TEST(AgradFwdVectorAppendRow,fd) {
  using stan::math::append_row;
  using stan::agrad::vector_fd;
  using Eigen::VectorXd;

  vector_fd a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;

  vector_fd ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
      EXPECT_EQ(a(i).d_, ab_append_row(i).d_);
  
  for (int i = 0; i < 7; i++)
      EXPECT_EQ(ab_append_row(i).val_, adb_append_row(i));
}

TEST(AgradFwdMatrixAppendRow,ffd) {
  using stan::math::append_row;
  using stan::agrad::matrix_ffd;
  using Eigen::MatrixXd;

  matrix_ffd a(2,2);
  MatrixXd ad(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  ad << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  a(0,0).d_ = 2.0;
  a(0,1).d_ = 3.0;
  a(1,0).d_ = 4.0;
  a(1,1).d_ = 5.0;

  matrix_ffd ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_.val(), ab_append_row(i ,j).d_.val());
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_.val(), adb_append_row(i ,j));
}

TEST(AgradFwdVectorAppendRow,ffd) {
  using stan::math::append_row;
  using stan::agrad::vector_ffd;
  using Eigen::VectorXd;

  vector_ffd a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;

  vector_ffd ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(a(i).d_.val(), ab_append_row(i).d_.val());
  
  for (int i = 0; i < 7; i++)
    EXPECT_EQ(ab_append_row(i).val_.val(), adb_append_row(i));
}


TEST(AgradFwdMatrixAppendRow,fv) {
  using stan::math::append_row;
  using stan::agrad::matrix_fv;
  using Eigen::MatrixXd;

  matrix_fv a(2,2);
  MatrixXd ad(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  ad << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  a(0,0).d_ = 2.0;
  a(0,1).d_ = 3.0;
  a(1,0).d_ = 4.0;
  a(1,1).d_ = 5.0;

  matrix_fv ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_.val(), ab_append_row(i ,j).d_.val());
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_.val(), adb_append_row(i ,j));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0,0).val_);
  vars.push_back(a(0,1).val_);
  vars.push_back(a(1,0).val_);
  vars.push_back(a(1,1).val_);

  std::vector<double> grads;
  ab_append_row(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdVectorAppendRow,fv) {
  using stan::math::append_row;
  using stan::agrad::vector_fv;
  using Eigen::VectorXd;

  vector_fv a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;

  vector_fv ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(a(i).d_.val(), ab_append_row(i).d_.val());
  
  for (int i = 0; i < 7; i++)
    EXPECT_EQ(ab_append_row(i).val_.val(), adb_append_row(i));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0).val_);
  vars.push_back(a(1).val_);
  vars.push_back(a(2).val_);
  vars.push_back(a(3).val_);

  std::vector<double> grads;
  ab_append_row(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdMatrixAppendRow,fv2) {
  using stan::math::append_row;
  using stan::agrad::matrix_fv;
  using Eigen::MatrixXd;

  matrix_fv a(2,2);
  MatrixXd ad(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  ad << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  a(0,0).d_ = 2.0;
  a(0,1).d_ = 3.0;
  a(1,0).d_ = 4.0;
  a(1,1).d_ = 5.0;

  matrix_fv ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_.val(), ab_append_row(i ,j).d_.val());
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_.val(), adb_append_row(i ,j));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0,0).val_);
  vars.push_back(a(0,1).val_);
  vars.push_back(a(1,0).val_);
  vars.push_back(a(1,1).val_);

  std::vector<double> grads;
  ab_append_row(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdVectorAppendRow,fv2) {
  using stan::math::append_row;
  using stan::agrad::vector_fv;
  using Eigen::VectorXd;

  vector_fv a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;

  vector_fv ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
      EXPECT_EQ(a(i).d_, ab_append_row(i).d_);
  
  for (int i = 0; i < 7; i++)
      EXPECT_EQ(ab_append_row(i).val_, adb_append_row(i));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0).val_);
  vars.push_back(a(1).val_);
  vars.push_back(a(2).val_);
  vars.push_back(a(3).val_);

  std::vector<double> grads;
  ab_append_row(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}


TEST(AgradFwdMatrixAppendRow,ffv1) {
  using stan::math::append_row;
  using stan::agrad::matrix_ffv;
  using Eigen::MatrixXd;

  matrix_ffv a(2,2);
  MatrixXd ad(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  ad << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  a(0,0).d_ = 2.0;
  a(0,1).d_ = 3.0;
  a(1,0).d_ = 4.0;
  a(1,1).d_ = 5.0;

  matrix_ffv ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_.val().val(), ab_append_row(i ,j).d_.val().val());
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_.val().val(), adb_append_row(i ,j));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0,0).val_.val_);
  vars.push_back(a(0,1).val_.val_);
  vars.push_back(a(1,0).val_.val_);
  vars.push_back(a(1,1).val_.val_);

  std::vector<double> grads;
  ab_append_row(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdVectorAppendRow,ffv1) {
  using stan::math::append_row;
  using stan::agrad::vector_ffv;
  using Eigen::VectorXd;

  vector_ffv a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;

  vector_ffv ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(a(i).d_.val().val(), ab_append_row(i).d_.val().val());
  
  for (int i = 0; i < 7; i++)
    EXPECT_EQ(ab_append_row(i).val_.val().val(), adb_append_row(i));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0).val_.val_);
  vars.push_back(a(1).val_.val_);
  vars.push_back(a(2).val_.val_);
  vars.push_back(a(3).val_.val_);

  std::vector<double> grads;
  ab_append_row(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdMatrixAppendRow,ffv2) {
  using stan::math::append_row;
  using stan::agrad::matrix_ffv;
  using Eigen::MatrixXd;

  matrix_ffv a(2,2);
  MatrixXd ad(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  ad << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  a(0,0).d_ = 2.0;
  a(0,1).d_ = 3.0;
  a(1,0).d_ = 4.0;
  a(1,1).d_ = 5.0;
  a(0,0).val_.d_ = 1.0;
  a(0,1).val_.d_ = 1.0;
  a(1,0).val_.d_ = 1.0;
  a(1,1).val_.d_ = 1.0;

  matrix_ffv ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_.val().val(), ab_append_row(i ,j).d_.val().val());
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_.val().val(), adb_append_row(i ,j));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0,0).val_.val_);
  vars.push_back(a(0,1).val_.val_);
  vars.push_back(a(1,0).val_.val_);
  vars.push_back(a(1,1).val_.val_);

  std::vector<double> grads;
  ab_append_row(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdVectorAppendRow,ffv2) {
  using stan::math::append_row;
  using stan::agrad::vector_ffv;
  using Eigen::VectorXd;

  vector_ffv a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;
  a(0).val_.d_ = 1.0;
  a(1).val_.d_ = 1.0;
  a(2).val_.d_ = 1.0;
  a(3).val_.d_ = 1.0;

  vector_ffv ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(a(i).d_.val().val(), ab_append_row(i).d_.val().val());
  
  for (int i = 0; i < 7; i++)
    EXPECT_EQ(ab_append_row(i).val_.val().val(), adb_append_row(i));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0).val_.val_);
  vars.push_back(a(1).val_.val_);
  vars.push_back(a(2).val_.val_);
  vars.push_back(a(3).val_.val_);

  std::vector<double> grads;
  ab_append_row(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdMatrixAppendRow,ffv3) {
  using stan::math::append_row;
  using stan::agrad::matrix_ffv;
  using Eigen::MatrixXd;

  matrix_ffv a(2,2);
  MatrixXd ad(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  ad << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  a(0,0).d_ = 2.0;
  a(0,1).d_ = 3.0;
  a(1,0).d_ = 4.0;
  a(1,1).d_ = 5.0;
  a(0,0).val_.d_ = 1.0;
  a(0,1).val_.d_ = 1.0;
  a(1,0).val_.d_ = 1.0;
  a(1,1).val_.d_ = 1.0;

  matrix_ffv ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_.val().val(), ab_append_row(i ,j).d_.val().val());
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_.val().val(), adb_append_row(i ,j));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0,0).val_.val_);
  vars.push_back(a(0,1).val_.val_);
  vars.push_back(a(1,0).val_.val_);
  vars.push_back(a(1,1).val_.val_);

  std::vector<double> grads;
  ab_append_row(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdVectorAppendRow,ffv3) {
  using stan::math::append_row;
  using stan::agrad::vector_ffv;
  using Eigen::VectorXd;

  vector_ffv a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;
  a(0).val_.d_ = 1.0;
  a(1).val_.d_ = 1.0;
  a(2).val_.d_ = 1.0;
  a(3).val_.d_ = 1.0;

  vector_ffv ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(a(i).d_.val().val(), ab_append_row(i).d_.val().val());
  
  for (int i = 0; i < 7; i++)
    EXPECT_EQ(ab_append_row(i).val_.val().val(), adb_append_row(i));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0).val_.val_);
  vars.push_back(a(1).val_.val_);
  vars.push_back(a(2).val_.val_);
  vars.push_back(a(3).val_.val_);

  std::vector<double> grads;
  ab_append_row(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdMatrixAppendRow,ffv4) {
  using stan::math::append_row;
  using stan::agrad::matrix_ffv;
  using Eigen::MatrixXd;

  matrix_ffv a(2,2);
  MatrixXd ad(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  ad << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  a(0,0).d_ = 2.0;
  a(0,1).d_ = 3.0;
  a(1,0).d_ = 4.0;
  a(1,1).d_ = 5.0;
  a(0,0).val_.d_ = 1.0;
  a(0,1).val_.d_ = 1.0;
  a(1,0).val_.d_ = 1.0;
  a(1,1).val_.d_ = 1.0;

  matrix_ffv ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_.val().val(), ab_append_row(i ,j).d_.val().val());
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_.val().val(), adb_append_row(i ,j));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0,0).val_.val_);
  vars.push_back(a(0,1).val_.val_);
  vars.push_back(a(1,0).val_.val_);
  vars.push_back(a(1,1).val_.val_);

  std::vector<double> grads;
  ab_append_row(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdVectorAppendRow,ffv4) {
  using stan::math::append_row;
  using stan::agrad::vector_ffv;
  using Eigen::VectorXd;

  vector_ffv a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;
  a(0).val_.d_ = 1.0;
  a(1).val_.d_ = 1.0;
  a(2).val_.d_ = 1.0;
  a(3).val_.d_ = 1.0;

  vector_ffv ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(a(i).d_.val().val(), ab_append_row(i).d_.val().val());
  
  for (int i = 0; i < 7; i++)
    EXPECT_EQ(ab_append_row(i).val_.val().val(), adb_append_row(i));

  std::vector<stan::agrad::var> vars;
  vars.push_back(a(0).val_.val_);
  vars.push_back(a(1).val_.val_);
  vars.push_back(a(2).val_.val_);
  vars.push_back(a(3).val_.val_);

  std::vector<double> grads;
  ab_append_row(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
