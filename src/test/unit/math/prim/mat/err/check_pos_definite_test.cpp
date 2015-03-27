#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/err/check_pos_definite.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>


const char* function = "function";
class ErrorHandlingMatrix : public ::testing::Test {
public:
  void SetUp() {
  }
  
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
};

TEST_F(ErrorHandlingMatrix, checkPosDefinite) {
  using stan::math::check_pos_definite;

  y.resize(1,1);
  y << 1;
  EXPECT_TRUE(check_pos_definite(function, "y", y));

  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt_1(y); 
  EXPECT_TRUE(check_pos_definite(function, "y", llt_1));

  Eigen::LDLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > ldlt_1 = y.ldlt(); 
  EXPECT_TRUE(check_pos_definite(function, "y", ldlt_1));

  y.resize(3,3);
  y << 
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;
  EXPECT_TRUE(check_pos_definite(function, "y", y));

  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt_2(y); 
  EXPECT_TRUE(check_pos_definite(function, "y", llt_2));

  Eigen::LDLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > ldlt_2 = y.ldlt(); 
  EXPECT_TRUE(check_pos_definite(function, "y", ldlt_2));
}

TEST_F(ErrorHandlingMatrix, checkPosDefinite_not_square) {
  using stan::math::check_pos_definite;
  std::stringstream expected_msg;
  
  y.resize(3, 4);
  expected_msg << "Expecting a square matrix; rows of y (3) and columns of y (4) must match in size";
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y),
                   std::invalid_argument,
                   expected_msg.str());
  y.resize(2,3);
  y <<
    1, 1, 
    1, 1,
    1, 1; 
  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt(y.rows()); 
  EXPECT_DEATH(llt.compute(y),"Assertion failed");

  EXPECT_DEATH(y.ldlt(), "Assertion failed"); 
}

TEST_F(ErrorHandlingMatrix, checkPosDefinite_0_size) {
  using stan::math::check_pos_definite;
  std::string expected_msg;

  expected_msg = "y must have a positive size, but is 0; dimension size expression = rows";
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y),
                   std::invalid_argument,
                   expected_msg);
  Eigen::MatrixXd x;
  x.resize(0,0);
  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt(x.rows()); 
  llt.compute(x);
  EXPECT_NO_THROW(check_pos_definite(function, "x", llt));
  Eigen::LDLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > ldlt(x.rows()); 
  EXPECT_DEATH(ldlt.compute(x),"");
}

TEST_F(ErrorHandlingMatrix, checkPosDefinite_non_symmetric) {
  using stan::math::check_pos_definite;
  std::string expected_msg;

  y.resize(3,3);
  y <<
    1, 0, 0,
    0, 1, 0.5,
    0, 0, 1;
  
  expected_msg = "y is not symmetric. y[2,3] = 0.5, but y[3,2] = 0";
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y),
                   std::domain_error,
                   expected_msg);

  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt(y); 
  EXPECT_NO_THROW(check_pos_definite(function, "y", llt));

  Eigen::LDLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > ldlt = y.ldlt(); 
  EXPECT_NO_THROW(check_pos_definite(function, "y", ldlt));
}

TEST_F(ErrorHandlingMatrix, checkPosDefinite_non_pos_definite) {
  using stan::math::check_pos_definite;
  std::stringstream expected_msg1_mat;
  std::stringstream expected_msg1_llt;
  std::stringstream expected_msg1_ldlt;
  std::stringstream expected_msg2_mat;
  std::stringstream expected_msg3_mat;
  std::stringstream expected_msg4;

  y.resize(3,3);
  y <<
    -1, 0, 0,
    0, -1, 0,
    0, 0, -1;
  
  expected_msg1_mat << "function: y is not positive definite:\n" << 
    "-1  0  0\n 0 -1  0\n 0  0 -1";
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y),
                   std::domain_error,
                   expected_msg1_mat.str());

  expected_msg1_llt << "function: Cholesky decomposition of y failed";
  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt_err1(y); 
  EXPECT_THROW_MSG(check_pos_definite(function, "y", llt_err1),
                   std::domain_error,
                   expected_msg1_llt.str());
  expected_msg1_ldlt << "function: LDLT decomposition of y failed";
  Eigen::LDLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > ldlt_err1 = y.ldlt(); 
  EXPECT_THROW_MSG(check_pos_definite(function, "y", ldlt_err1),
                   std::domain_error,
                   expected_msg1_ldlt.str());
  y.resize(2,2);
  y <<
    1, 2, 
    2, 1; 
  expected_msg2_mat << "function: y is not positive definite:\n" <<
    "1 2\n2 1";
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y),
                   std::domain_error,
                   expected_msg2_mat.str());
  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt_err2(y); 
  EXPECT_THROW_MSG(check_pos_definite(function, "y", llt_err2),
                   std::domain_error,
                   expected_msg1_llt.str());

  Eigen::LDLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > ldlt_err2 = y.ldlt(); 
  EXPECT_THROW_MSG(check_pos_definite(function, "y", ldlt_err2),
                   std::domain_error,
                   expected_msg1_ldlt.str());
  y <<
    1, 1, 
    1, 1; 
  expected_msg3_mat << "function: y is not positive definite:\n" <<
    "1 1\n1 1";
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y),
                   std::domain_error,
                   expected_msg3_mat.str());

  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt_err3(y); 
  EXPECT_THROW_MSG(check_pos_definite(function, "y", llt_err3),
                   std::domain_error,
                   expected_msg1_llt.str());
  Eigen::LDLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > ldlt_err3 = y.ldlt(); 
  EXPECT_THROW_MSG(check_pos_definite(function, "y", ldlt_err3),
                   std::domain_error,
                   expected_msg1_ldlt.str());
}

TEST_F(ErrorHandlingMatrix, checkPosDefinite_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  using stan::math::check_pos_definite;

  y.resize(1,1);
  y << nan;

  std::stringstream expected_msg;
  expected_msg << "function: y is not positive definite: "
               << nan;
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y), 
                   std::domain_error,
                   expected_msg.str());

  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt_err1(y); 
  EXPECT_THROW(check_pos_definite(function, "y", llt_err1),std::domain_error);

  Eigen::LDLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > ldlt_err1 = y.ldlt(); 
  EXPECT_THROW(check_pos_definite(function, "y", ldlt_err1),std::domain_error);
  
  y.resize(3,3);
  y << 2, -1, 0,
    -1, 2, -1,
    0, -1, 2;
  EXPECT_TRUE(check_pos_definite(function, 
                                 "y", y));
  for (int i = 0; i < y.rows(); i++)
    for (int j = 0; j < y.cols(); j++) {
      y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
      y(i,j) = nan;
      if (i >= j) {
        expected_msg.str("");
        if (i == j)
          expected_msg << "function: y["
                       << j*y.cols() + i + 1 
                       << "] is " << nan
                       << ", but must not be nan!";
        else
          expected_msg << "function: y is not symmetric. " 
                       << "y[" << j+1 << "," << i+1 << "] = " << y(j,i)
                       << ", but y[" << i+1 << "," << j+1 << "] = " << y(i,j);
        EXPECT_THROW_MSG(check_pos_definite(function, "y", y), 
                         std::domain_error,
                         expected_msg.str());
      }
    }
  
  y << 2, -1, nan,
    -1, 2, -1,
    nan, -1, nan;
  Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > llt_err2(y); 
  EXPECT_THROW(check_pos_definite(function, "y", llt_err2),std::domain_error);

  Eigen::LDLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > ldlt_err2 = y.ldlt(); 
  EXPECT_THROW(check_pos_definite(function, "y", ldlt_err2),std::domain_error);

  y << 0, 0, 0, 0, 0, 0, 0, 0, 0;
  expected_msg.str("");
  expected_msg << "function: y is not positive definite:\n"
               << y;
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y), 
                   std::domain_error,
                   expected_msg.str());
}

