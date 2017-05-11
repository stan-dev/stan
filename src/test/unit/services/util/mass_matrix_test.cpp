#include <stan/services/util/mass_matrix.hpp>
#include <gtest/gtest.h>

TEST(mass_matrix, create_diag_sz1) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_diag_mass_matrix(1);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> diag_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(1,diag_vals.size());
  ASSERT_NEAR(1.0, diag_vals[0], 0.0001);
}

TEST(mass_matrix, create_diag_sz0) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_diag_mass_matrix(0);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> diag_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(0,diag_vals.size());
}

TEST(mass_matrix, create_diag_sz100) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_diag_mass_matrix(100);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> diag_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(100,diag_vals.size());
  ASSERT_NEAR(1.0, diag_vals[0], 0.0001);
  ASSERT_NEAR(1.0, diag_vals[99], 0.0001);
}

TEST(mass_matrix, create_dense_sz2) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_dense_mass_matrix(2);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> dense_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(4,dense_vals.size());
  ASSERT_NEAR(1.0, dense_vals[0], 0.0001);
  ASSERT_NEAR(1.0, dense_vals[3], 0.0001);
}

TEST(mass_matrix, create_dense_sz3) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_dense_mass_matrix(3);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> dense_vals
    = inv_mass_matrix.vals_r("mass_matrix");
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

TEST(mass_matrix, create_dense_sz10) {
  stan::io::dump dmp = 
    stan::services::util::create_unit_e_dense_mass_matrix(10);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> dense_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(100,dense_vals.size());
  ASSERT_NEAR(1.0, dense_vals[0], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[1], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[98], 0.0001);
  ASSERT_NEAR(1.0, dense_vals[99], 0.0001);
}

TEST(mass_matrix, read_diag_OK) {
  stan::callbacks::writer writer;
  std::string txt =
    "mass_matrix <- structure(c(0.787405, 0.884987, 1.19869),.Dim=c(3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  Eigen::VectorXd inv_mass_matrix = 
    stan::services::util::read_diag_mass_matrix(dump, 3, writer);
  EXPECT_EQ(3,inv_mass_matrix.size());
  ASSERT_NEAR(0.787405, inv_mass_matrix(0), 0.000001);
}  

TEST(mass_matrix, read_diag_bad1) {
  stan::callbacks::writer writer;
  std::string txt =
    "bad_name <- structure(c(0.787405, 0.884987, 1.19869),.Dim=c(3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_THROW(stan::services::util::read_diag_mass_matrix(dump, 3, writer),
               std::domain_error);
}  

TEST(mass_matrix, read_diag_bad2) {
  stan::callbacks::writer writer;
  std::string txt =
    "mass_matrix <- structure(c("
    " 0.926739, 0.0734898, -0.12395, "
    " 0.0734898, 0.876038, -0.051543, "
    " -0.12395, -0.051543, 0.8274 "
    "), .Dim  = c(3,3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_THROW(stan::services::util::read_diag_mass_matrix(dump, 9, writer),
               std::exception);
}  

TEST(mass_matrix, read_dense_OK) {
  stan::callbacks::writer writer;
  std::string txt =
    "mass_matrix <- structure(c("
    " 0.926739, 0.0734898, -0.12395, "
    " 0.0734898, 0.876038, -0.051543, "
    " -0.12395, -0.051543, 0.8274"
    "), .Dim  = c(3,3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  Eigen::MatrixXd inv_mass_matrix = 
    stan::services::util::read_dense_mass_matrix(dump, 3, writer);
  EXPECT_EQ(9,inv_mass_matrix.size());
  EXPECT_EQ(3,inv_mass_matrix.rows());
  EXPECT_EQ(3,inv_mass_matrix.cols());
  ASSERT_NEAR(0.926739, inv_mass_matrix(0), 0.000001);
  ASSERT_NEAR(0.0734898, inv_mass_matrix(3), 0.000001);
  ASSERT_NEAR(0.8274, inv_mass_matrix(8), 0.000001);
}  

TEST(mass_matrix, read_dense_bad1) {
  stan::callbacks::writer writer;
  std::string txt =
    "wrong_name <- structure(c("
    " 0.926739, 0.0734898, -0.12395, "
    " 0.0734898, 0.876038, -0.051543, "
    " -0.12395, -0.051543, 0.8274"
    "), .Dim  = c(3,3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_THROW(stan::services::util::read_dense_mass_matrix(dump, 3, writer),
               std::domain_error);
}  

TEST(mass_matrix, read_dense_bad2) {
  stan::callbacks::writer writer;
  std::string txt =
    "mass_matrix <- structure(c("
    " 0.926739, 0.0734898, -0.12395, "
    " 0.0734898, 0.876038, -0.051543, "
    " -0.12395, -0.051543, 0.8274 "
    "), .Dim  = c(9))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_THROW(stan::services::util::read_dense_mass_matrix(dump, 3, writer),
               std::domain_error);
}  

TEST(mass_matrix, validate_diag_imm) {
  stan::callbacks::writer writer;
  Eigen::VectorXd v1(1);
  v1(0) = 0.0;
  EXPECT_THROW(stan::services::util::validate_diag_mass_matrix(v1, writer),
               std::domain_error);
  v1(0) = 1.0;
  EXPECT_NO_THROW(stan::services::util::validate_diag_mass_matrix(v1, writer));

  Eigen::VectorXd v2(2);
  v2(0) = 1.0;
  v2(1) = 0.0;
  EXPECT_THROW(stan::services::util::validate_diag_mass_matrix(v2, writer),
               std::domain_error);

  v2(1) = -1.0;
  EXPECT_THROW(stan::services::util::validate_diag_mass_matrix(v2, writer),
               std::domain_error);

  v2(1) = 1.0;
  EXPECT_NO_THROW(stan::services::util::validate_diag_mass_matrix(v2, writer));
}


TEST(mass_matrix, validate_dense_imm) {
  stan::callbacks::writer writer;
  Eigen::MatrixXd m1(1,1);
  m1(0,0) = 0.0;
  EXPECT_THROW(stan::services::util::validate_dense_mass_matrix(m1, writer),
               std::domain_error);
  m1(0,0) = 1.0;
  EXPECT_NO_THROW(stan::services::util::validate_dense_mass_matrix(m1, writer));

  Eigen::MatrixXd m2(2,2);
  m2(0,0) = 1.0;
  m2(0,1) = 0.0;
  m2(1,0) = 0.0;
  m2(1,1) = 0.0;
  EXPECT_THROW(stan::services::util::validate_dense_mass_matrix(m2, writer),
               std::domain_error);

  m2(1,1) = 1.0;
  EXPECT_NO_THROW(stan::services::util::validate_dense_mass_matrix(m2, writer));


  m2(1,1) = -3.0;
  EXPECT_THROW(stan::services::util::validate_dense_mass_matrix(m2, writer),
               std::domain_error);
}
  
