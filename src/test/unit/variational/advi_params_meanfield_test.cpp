#include <stan/variational/families/normal_meanfield.hpp>
#include <vector>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(normal_meanfield_test, zero_init) {
  int my_dimension =  10;

  stan::variational::normal_meanfield my_normal_meanfield(my_dimension);
  EXPECT_FLOAT_EQ(my_dimension, my_normal_meanfield.dimension());

  const Eigen::VectorXd& mu_out    = my_normal_meanfield.mu();
  const Eigen::VectorXd& omega_out = my_normal_meanfield.omega();

  for (int i = 0; i < my_dimension; ++i) {
    EXPECT_FLOAT_EQ(0.0, mu_out(i));
    EXPECT_FLOAT_EQ(0.0, omega_out(i));
  }
}

TEST(normal_meanfield_test, dimension) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d omega;
  omega << -0.42, 0.8922, 13.4;

  stan::variational::normal_meanfield my_normal_meanfield(mu, omega);

  EXPECT_FLOAT_EQ(mu.size(), my_normal_meanfield.dimension());
  EXPECT_FLOAT_EQ(omega.size(), my_normal_meanfield.dimension());
}

TEST(normal_meanfield_test, mean_vector) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d omega;
  omega << -0.42, 0.8922, 13.4;

  stan::variational::normal_meanfield my_normal_meanfield(mu, omega);

  const Eigen::Vector3d& mu_out = my_normal_meanfield.mu();

  for (int i = 0; i < my_normal_meanfield.dimension(); ++i)
    EXPECT_FLOAT_EQ(mu(i), mu_out(i));

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d mu_nan = Eigen::VectorXd::Constant(3, nan);
  EXPECT_THROW(stan::variational::normal_meanfield my_normal_meanfield_nan(mu_nan, omega),
                   std::domain_error);
  EXPECT_THROW(my_normal_meanfield.set_mu(mu_nan),
                   std::domain_error);
  Eigen::Vector3d omega_nan = Eigen::VectorXd::Constant(3,nan);
  EXPECT_THROW(stan::variational::normal_meanfield my_normal_meanfield_nan(mu, omega_nan);,
                   std::domain_error);

  my_normal_meanfield.set_to_zero();
  const Eigen::Vector3d& mu_out_zero = my_normal_meanfield.mu();

  for (int i = 0; i < my_normal_meanfield.dimension(); ++i) {
    EXPECT_FLOAT_EQ(0.0, mu_out_zero(i));
  }

}

TEST(normal_meanfield_test, omega_vector) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d omega;
  omega << -0.42, 0.8922, 13.4;

  stan::variational::normal_meanfield my_normal_meanfield(mu, omega);

  const Eigen::Vector3d& omega_out = my_normal_meanfield.omega();

  for (int i = 0; i < my_normal_meanfield.dimension(); ++i)
    EXPECT_FLOAT_EQ(omega(i), omega_out(i));

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d omega_nan = Eigen::VectorXd::Constant(3, nan);

  EXPECT_THROW(my_normal_meanfield.set_omega(omega_nan);,
                   std::domain_error);

  my_normal_meanfield.set_to_zero();
  const Eigen::Vector3d& omega_out_zero = my_normal_meanfield.omega();

  for (int i = 0; i < my_normal_meanfield.dimension(); ++i) {
    EXPECT_FLOAT_EQ(0.0, omega_out_zero(i));
  }
}

TEST(normal_meanfield_test, entropy) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d omega;
  omega << -0.42, 0.8922, 13.4;

  double entropy_true = 18.129015599614018;

  stan::variational::normal_meanfield my_normal_meanfield(mu, omega);

  const double entropy_out = my_normal_meanfield.entropy();

  EXPECT_FLOAT_EQ(entropy_out, entropy_true);

}

TEST(normal_meanfield_test, transform) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d omega;
  omega << -0.42, 0.8922, 13.4;

  stan::variational::normal_meanfield my_normal_meanfield(mu, omega);

  Eigen::Vector3d x;
  x << 7.1, -9.2, 0.59;

  Eigen::Vector3d x_transformed;
  x_transformed << 1.036503242068690e01,  -2.565253407151558e01,
         3.894020358120325e05;

  Eigen::Vector3d x_result;
  x_result = my_normal_meanfield.transform(x);

  for (int i = 0; i < my_normal_meanfield.dimension(); ++i)
    EXPECT_FLOAT_EQ(x_result(i), x_transformed(i));

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d x_nan = Eigen::VectorXd::Constant(3, nan);

  EXPECT_THROW(my_normal_meanfield.transform(x_nan);,
                   std::domain_error);
}
