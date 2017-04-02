#include <stan/services/util/create_ident_diag_mass_matrix.hpp>
#include <stan/services/util/create_ident_dense_mass_matrix.hpp>
#include <gtest/gtest.h>

TEST(inv_mass_matrix, diag_sz1) {
  stan::io::dump dmp = 
    stan::services::util::create_ident_diag_mass_matrix(1);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> diag_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(1,diag_vals.size());
  ASSERT_NEAR(1.0, diag_vals[0], 0.0001);
}

TEST(inv_mass_matrix, diag_sz0) {
  stan::io::dump dmp = 
    stan::services::util::create_ident_diag_mass_matrix(0);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> diag_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(0,diag_vals.size());
}

TEST(inv_mass_matrix, diag_sz100) {
  stan::io::dump dmp = 
    stan::services::util::create_ident_diag_mass_matrix(100);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> diag_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(100,diag_vals.size());
  ASSERT_NEAR(1.0, diag_vals[0], 0.0001);
  ASSERT_NEAR(1.0, diag_vals[99], 0.0001);
}

TEST(inv_mass_matrix, dense_sz2) {
  stan::io::dump dmp = 
    stan::services::util::create_ident_dense_mass_matrix(2);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> dense_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(4,dense_vals.size());
  ASSERT_NEAR(1.0, dense_vals[0], 0.0001);
  ASSERT_NEAR(1.0, dense_vals[3], 0.0001);
}

TEST(inv_mass_matrix, dense_sz3) {
  stan::io::dump dmp = 
    stan::services::util::create_ident_dense_mass_matrix(3);
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

TEST(inv_mass_matrix, dense_sz10) {
  stan::io::dump dmp = 
    stan::services::util::create_ident_dense_mass_matrix(10);
  stan::io::var_context& inv_mass_matrix = dmp;
  std::vector<double> dense_vals
    = inv_mass_matrix.vals_r("mass_matrix");
  EXPECT_EQ(100,dense_vals.size());
  ASSERT_NEAR(1.0, dense_vals[0], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[1], 0.0001);
  ASSERT_NEAR(0.0, dense_vals[98], 0.0001);
  ASSERT_NEAR(1.0, dense_vals[99], 0.0001);
}
