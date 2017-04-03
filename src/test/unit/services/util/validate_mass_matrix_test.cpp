#include <stan/services/util/validate_diag_mass_matrix.hpp>
#include <stan/services/util/validate_dense_mass_matrix.hpp>
#include <stan/callbacks/writer.hpp>
#include <Eigen/Dense>
#include <sstream>
#include <gtest/gtest.h>

TEST(inv_mass_matrix, diag_imm) {
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


TEST(inv_mass_matrix, dense_imm) {
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
