#include <stan/services/util/inv_metric.hpp>
#include <gtest/gtest.h>

TEST(inv_metric, create_diag_sz1) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_diag_inv_metric(1);
  stan::io::var_context& inv_inv_metric = dmp;
  std::vector<double> diag_vals
    = inv_inv_metric.vals_r("inv_metric");
  EXPECT_EQ(1,diag_vals.size());
  ASSERT_NEAR(1.0, diag_vals[0], 0.0001);
}

TEST(inv_metric, create_diag_sz0) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_diag_inv_metric(0);
  stan::io::var_context& inv_inv_metric = dmp;
  std::vector<double> diag_vals
    = inv_inv_metric.vals_r("inv_metric");
  EXPECT_EQ(0,diag_vals.size());
}

TEST(inv_metric, create_diag_sz100) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_diag_inv_metric(100);
  stan::io::var_context& inv_inv_metric = dmp;
  std::vector<double> diag_vals
    = inv_inv_metric.vals_r("inv_metric");
  EXPECT_EQ(100,diag_vals.size());
  ASSERT_NEAR(1.0, diag_vals[0], 0.0001);
  ASSERT_NEAR(1.0, diag_vals[99], 0.0001);
}

TEST(inv_metric, create_dense_sz2) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_dense_inv_metric(2);
  stan::io::var_context& inv_inv_metric = dmp;
  std::vector<double> dense_vals
    = inv_inv_metric.vals_r("inv_metric");
  EXPECT_EQ(4,dense_vals.size());
  ASSERT_NEAR(1.0, dense_vals[0], 0.0001);
  ASSERT_NEAR(1.0, dense_vals[3], 0.0001);
}

TEST(inv_metric, create_dense_sz3) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_dense_inv_metric(3);
  stan::io::var_context& inv_inv_metric = dmp;
  std::vector<double> dense_vals
    = inv_inv_metric.vals_r("inv_metric");
  EXPECT_EQ(9,dense_vals.size());
  ASSERT_NEAR(1.0, dense_vals[0], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[1], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[2], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[3], 0.0001);
  ASSERT_NEAR(1.0, dense_vals[4], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[5], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[6], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[7], 0.0001);
  ASSERT_NEAR(1.0, dense_vals[8], 0.0001);
}

TEST(inv_metric, create_dense_sz10) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_dense_inv_metric(10);
  stan::io::var_context& inv_inv_metric = dmp;
  std::vector<double> dense_vals
    = inv_inv_metric.vals_r("inv_metric");
  EXPECT_EQ(100,dense_vals.size());
  ASSERT_NEAR(1.0, dense_vals[0], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[1], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[98], 0.0001);
  ASSERT_NEAR(1.0, dense_vals[99], 0.0001);
}

TEST(inv_metric, read_diag_OK) {
  stan::callbacks::logger logger;
  std::string txt =
    "inv_metric <- structure(c(0.787405, 0.884987, 1.19869),.Dim=c(3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  Eigen::VectorXd inv_inv_metric = 
    stan::services::util::read_diag_inv_metric(dump, 3, logger);
  EXPECT_EQ(3,inv_inv_metric.size());
  ASSERT_NEAR(0.787405, inv_inv_metric(0), 0.000001);
}  

TEST(inv_metric, read_diag_bad1) {
  stan::callbacks::logger logger;
  std::string txt =
    "bad_name <- structure(c(0.787405, 0.884987, 1.19869),.Dim=c(3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_THROW(stan::services::util::read_diag_inv_metric(dump, 3, logger),
               std::domain_error);
}  

TEST(inv_metric, read_diag_bad2) {
  stan::callbacks::logger logger;
  std::string txt =
    "inv_metric <- structure(c("
    " 0.926739, 0.0734898, -0.12395, "
    " 0.0734898, 0.876038, -0.051543, "
    " -0.12395, -0.051543, 0.8274 "
    "), .Dim  = c(3,3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_THROW(stan::services::util::read_diag_inv_metric(dump, 9, logger),
               std::exception);
}  

TEST(inv_metric, read_dense_OK) {
  stan::callbacks::logger logger;
  std::string txt =
    "inv_metric <- structure(c("
    " 0.926739, 0.0734898, -0.12395, "
    " 0.0734898, 0.876038, -0.051543, "
    " -0.12395, -0.051543, 0.8274"
    "), .Dim  = c(3,3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  Eigen::MatrixXd inv_inv_metric = 
    stan::services::util::read_dense_inv_metric(dump, 3, logger);
  EXPECT_EQ(9,inv_inv_metric.size());
  EXPECT_EQ(3,inv_inv_metric.rows());
  EXPECT_EQ(3,inv_inv_metric.cols());
  ASSERT_NEAR(0.926739, inv_inv_metric(0), 0.000001);
  ASSERT_NEAR(0.0734898, inv_inv_metric(3), 0.000001);
  ASSERT_NEAR(0.8274, inv_inv_metric(8), 0.000001);
}  

TEST(inv_metric, read_dense_bad1) {
  stan::callbacks::logger logger;
  std::string txt =
    "wrong_name <- structure(c("
    " 0.926739, 0.0734898, -0.12395, "
    " 0.0734898, 0.876038, -0.051543, "
    " -0.12395, -0.051543, 0.8274"
    "), .Dim  = c(3,3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_THROW(stan::services::util::read_dense_inv_metric(dump, 3, logger),
               std::domain_error);
}  

TEST(inv_metric, read_dense_bad2) {
  stan::callbacks::logger logger;
  std::string txt =
    "inv_metric <- structure(c("
    " 0.926739, 0.0734898, -0.12395, "
    " 0.0734898, 0.876038, -0.051543, "
    " -0.12395, -0.051543, 0.8274 "
    "), .Dim  = c(9))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_THROW(stan::services::util::read_dense_inv_metric(dump, 3, logger),
               std::domain_error);
}  

TEST(inv_metric, validate_diag_imm) {
  stan::callbacks::logger logger;
  Eigen::VectorXd v1(1);
  v1(0) = 0.0;
  EXPECT_THROW(stan::services::util::validate_diag_inv_metric(v1, logger),
               std::domain_error);
  v1(0) = 1.0;
  EXPECT_NO_THROW(stan::services::util::validate_diag_inv_metric(v1, logger));

  Eigen::VectorXd v2(2);
  v2(0) = 1.0;
  v2(1) = 0.0;
  EXPECT_THROW(stan::services::util::validate_diag_inv_metric(v2, logger),
               std::domain_error);

  v2(1) = -1.0;
  EXPECT_THROW(stan::services::util::validate_diag_inv_metric(v2, logger),
               std::domain_error);

  v2(1) = 1.0;
  EXPECT_NO_THROW(stan::services::util::validate_diag_inv_metric(v2, logger));
}


TEST(inv_metric, validate_dense_imm) {
  stan::callbacks::logger logger;
  Eigen::MatrixXd m1(1,1);
  m1(0,0) = 0.0;
  EXPECT_THROW(stan::services::util::validate_dense_inv_metric(m1, logger),
               std::domain_error);
  m1(0,0) = 1.0;
  EXPECT_NO_THROW(stan::services::util::validate_dense_inv_metric(m1, logger));

  Eigen::MatrixXd m2(2,2);
  m2(0,0) = 1.0;
  m2(0,1) = 0.0;
  m2(1,0) = 0.0;
  m2(1,1) = 0.0;
  EXPECT_THROW(stan::services::util::validate_dense_inv_metric(m2, logger),
               std::domain_error);

  m2(1,1) = 1.0;
  EXPECT_NO_THROW(stan::services::util::validate_dense_inv_metric(m2, logger));


  m2(1,1) = -3.0;
  EXPECT_THROW(stan::services::util::validate_dense_inv_metric(m2, logger),
               std::domain_error);
}
  
