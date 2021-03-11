#include <stan/io/deserializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

// lb

TEST(deserializer_vector, lb_free) {
  double lb = 0.5;
  Eigen::VectorXd uvec1 = Eigen::VectorXd::Random(3);
  Eigen::VectorXd uvec2 = Eigen::VectorXd::Random(3);
  Eigen::VectorXd vec_bad(2);
  Eigen::VectorXd vec1 = stan::math::lb_constrain(uvec1, lb);
  Eigen::VectorXd vec2 = stan::math::lb_constrain(uvec2, lb);
  vec_bad << -0.5, 1.0;

  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::VectorXd> std_vec = { vec1, vec2 },
    std_vec_bad = { vec1, vec_bad };

  stan::test::expect_near_rel("deserializer free",
			      uvec1,
			      deserializer.free_lb(vec1, lb));
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_lb(std_vec, lb);
  stan::test::expect_near_rel("deserializer free",
			      uvec1,
			      std_uvec[0]);
  stan::test::expect_near_rel("deserializer free",
			      uvec2,
			      std_uvec[1]);

  EXPECT_THROW(deserializer.free_lb(vec_bad, lb), std::domain_error);
  EXPECT_THROW(deserializer.free_lb(std_vec_bad, lb), std::domain_error);
}

// ub

TEST(deserializer_vector, ub_free) {
  double ub = 0.5;
  Eigen::VectorXd uvec1 = Eigen::VectorXd::Random(3);
  Eigen::VectorXd uvec2 = Eigen::VectorXd::Random(3);
  Eigen::VectorXd vec_bad(2);
  Eigen::VectorXd vec1 = stan::math::ub_constrain(uvec1, ub);
  Eigen::VectorXd vec2 = stan::math::ub_constrain(uvec2, ub);
  vec_bad << -0.5, 1.0;

  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::VectorXd> std_vec = { vec1, vec2 },
    std_vec_bad = { vec1, vec_bad };

  stan::test::expect_near_rel("deserializer free",
			      uvec1,
			      deserializer.free_ub(vec1, ub));
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_ub(std_vec, ub);
  stan::test::expect_near_rel("deserializer free",
			      uvec1,
			      std_uvec[0]);
  stan::test::expect_near_rel("deserializer free",
			      uvec2,
			      std_uvec[1]);

  EXPECT_THROW(deserializer.free_ub(vec_bad, ub), std::domain_error);
  EXPECT_THROW(deserializer.free_ub(std_vec_bad, ub), std::domain_error);
}

// lub

TEST(deserializer_vector, lub_free) {
  double lb = 0.2;
  double ub = 0.5;
  Eigen::VectorXd uvec1 = Eigen::VectorXd::Random(3);
  Eigen::VectorXd uvec2 = Eigen::VectorXd::Random(3);
  Eigen::VectorXd vec_bad(2);
  Eigen::VectorXd vec1 = stan::math::lub_constrain(uvec1, lb, ub);
  Eigen::VectorXd vec2 = stan::math::lub_constrain(uvec2, lb, ub);
  vec_bad << -0.5, 1.0;

  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::VectorXd> std_vec = { vec1, vec2 },
    std_vec_bad = { vec1, vec_bad };

  stan::test::expect_near_rel("deserializer free",
			      uvec1,
			      deserializer.free_lub(vec1, lb, ub));
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_lub(std_vec, lb, ub);
  stan::test::expect_near_rel("deserializer free",
			      uvec1,
			      std_uvec[0]);
  stan::test::expect_near_rel("deserializer free",
			      uvec2,
			      std_uvec[1]);

  EXPECT_THROW(deserializer.free_lub(vec_bad, lb, ub), std::domain_error);
  EXPECT_THROW(deserializer.free_lub(std_vec_bad, lb, ub), std::domain_error);
}

// offset multiplier

TEST(deserializer_vector, offset_multiplier_free) {
  double offset = 0.2;
  double multiplier = 0.5;
  Eigen::VectorXd uvec1 = Eigen::VectorXd::Random(3);
  Eigen::VectorXd uvec2 = Eigen::VectorXd::Random(3);
  Eigen::VectorXd vec_bad(2);
  Eigen::VectorXd vec1 = stan::math::offset_multiplier_constrain(uvec1, offset, multiplier);
  Eigen::VectorXd vec2 = stan::math::offset_multiplier_constrain(uvec2, offset, multiplier);
  vec_bad << -0.5, 1.0;

  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::VectorXd> std_vec = { vec1, vec2 },
    std_vec_bad = { vec1, vec_bad };

  stan::test::expect_near_rel("deserializer free",
			      uvec1,
			      deserializer.free_offset_multiplier(vec1, offset, multiplier));
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_offset_multiplier(std_vec, offset, multiplier);
  stan::test::expect_near_rel("deserializer free",
			      uvec1,
			      std_uvec[0]);
  stan::test::expect_near_rel("deserializer free",
			      uvec2,
			      std_uvec[1]);
}

// unit vector

TEST(deserializer_vector, unit_vector_free) {
  Eigen::VectorXd vec1(3), vec2(3), vec_bad(2);
  vec1 << 0.2, 0.2, std::sqrt(1.0 - 0.2 * 0.2 - 0.2 * 0.2);
  vec2 << 0.1, 0.4, std::sqrt(1.0 - 0.1 * 0.1 - 0.4 * 0.4);
  vec_bad << -0.5, 1.0;

  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::VectorXd> std_vec = { vec1, vec2 },
    std_vec_bad = { vec1, vec_bad };

  Eigen::VectorXd uvec = deserializer.free_unit_vector(vec1);
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_unit_vector(std_vec);

  EXPECT_THROW(deserializer.free_unit_vector(vec_bad), std::domain_error);
  EXPECT_THROW(deserializer.free_unit_vector(std_vec_bad), std::domain_error);
}

// simplex

TEST(deserializer_vector, simplex_free) {
  Eigen::VectorXd vec1(3), vec2(3), vec_bad(2);
  vec1 << 0.2, 0.2, 1.0 - 0.2 - 0.2;
  vec2 << 0.1, 0.4, 1.0 - 0.1 - 0.4;
  vec_bad << -0.5, 1.0;

  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::VectorXd> std_vec = { vec1, vec2 },
    std_vec_bad = { vec1, vec_bad };

  Eigen::VectorXd uvec = deserializer.free_simplex(vec1);
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_simplex(std_vec);

  EXPECT_THROW(deserializer.free_simplex(vec_bad), std::domain_error);
  EXPECT_THROW(deserializer.free_simplex(std_vec_bad), std::domain_error);
}

// ordered

TEST(deserializer_vector, ordered_free) {
  Eigen::VectorXd vec1(3), vec2(3), vec_bad(2);
  vec1 << -0.2, 0.3, 1.0;
  vec2 << -1, -0.5, 1.0;
  vec_bad << 0.5, -1.0;

  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::VectorXd> std_vec = { vec1, vec2 },
    std_vec_bad = { vec1, vec_bad };

  Eigen::VectorXd uvec = deserializer.free_ordered(vec1);
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_ordered(std_vec);

  EXPECT_THROW(deserializer.free_ordered(vec_bad), std::domain_error);
  EXPECT_THROW(deserializer.free_ordered(std_vec_bad), std::domain_error);
}

// positive_ordered

TEST(deserializer_vector, positive_ordered_free) {
  Eigen::VectorXd vec1(3), vec2(3), vec_bad(2);
  vec1 << 0.1, 0.2, 0.7;
  vec2 << 0.1, 0.4, 0.5;
  vec_bad << -0.5, 1.0;

  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::VectorXd> std_vec = { vec1, vec2 },
    std_vec_bad = { vec1, vec_bad };

  Eigen::VectorXd uvec = deserializer.free_positive_ordered(vec1);
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_positive_ordered(std_vec);

  EXPECT_THROW(deserializer.free_positive_ordered(vec_bad), std::domain_error);
  EXPECT_THROW(deserializer.free_positive_ordered(std_vec_bad), std::domain_error);
}

// cholesky_factor_cov

TEST(deserializer_vector, cholesky_factor_free) {
  int M = 3;
  int N = 2;
  int P1 = (N * (N + 1)) / 2 + (M - N) * N;
  int P2 = (M * (M + 1)) / 2;
  Eigen::VectorXd vec1 = Eigen::VectorXd::Random(P1);
  Eigen::VectorXd vec2 = Eigen::VectorXd::Random(P2);

  Eigen::MatrixXd L1 = stan::math::cholesky_factor_constrain(vec1, M, N);
  Eigen::MatrixXd L2 = stan::math::cholesky_factor_constrain(vec2, M, M);
  Eigen::MatrixXd L_bad = stan::math::add_diag(L1, -10.0);
  
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::MatrixXd> std_vec = { L1, L2 },
    std_vec_bad = { L1, L_bad };

  stan::test::expect_near_rel("deserializer_free",
				    vec1,
				    deserializer.free_cholesky_factor_cov(L1));
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_cholesky_factor_cov(std_vec);

  stan::test::expect_near_rel("deserializer_free",
				    vec1,
				    std_uvec[0]);
  stan::test::expect_near_rel("deserializer_free",
				    vec2,
				    std_uvec[1]);

  EXPECT_THROW(deserializer.free_cholesky_factor_cov(L_bad), std::domain_error);
  EXPECT_THROW(deserializer.free_cholesky_factor_cov(std_vec_bad), std::domain_error);
}

// cholesky_factor_corr

TEST(deserializer_vector, cholesky_corr_free) {
  int M = 3;
  int P = (M * (M - 1)) / 2;
  Eigen::VectorXd vec1 = Eigen::VectorXd::Random(P);
  Eigen::VectorXd vec2 = Eigen::VectorXd::Random(P);

  Eigen::MatrixXd L1 = stan::math::cholesky_corr_constrain(vec1, M);
  Eigen::MatrixXd L2 = stan::math::cholesky_corr_constrain(vec2, M);
  Eigen::MatrixXd L_bad(2, 3);
  
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::MatrixXd> std_vec = { L1, L2 },
    std_vec_bad = { L1, L_bad };

  stan::test::expect_near_rel("deserializer_free",
				    vec1,
				    deserializer.free_cholesky_factor_corr(L1));
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_cholesky_factor_corr(std_vec);

  stan::test::expect_near_rel("deserializer_free",
				    vec1,
				    std_uvec[0]);
  stan::test::expect_near_rel("deserializer_free",
				    vec2,
				    std_uvec[1]);

  EXPECT_THROW(deserializer.free_cholesky_factor_corr(L_bad), std::invalid_argument);
  EXPECT_THROW(deserializer.free_cholesky_factor_corr(std_vec_bad), std::invalid_argument);
}

// cov_matrix

TEST(deserializer_vector, cov_matrix_free) {
  int M = 3;
  int P = M + (M * (M - 1)) / 2;
  Eigen::VectorXd vec1 = Eigen::VectorXd::Random(P);
  Eigen::VectorXd vec2 = Eigen::VectorXd::Random(P);

  Eigen::MatrixXd L1 = stan::math::cov_matrix_constrain(vec1, M);
  Eigen::MatrixXd L2 = stan::math::cov_matrix_constrain(vec2, M);
  Eigen::MatrixXd L_bad(2, 3);
  
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::MatrixXd> std_vec = { L1, L2 },
    std_vec_bad = { L1, L_bad };

  stan::test::expect_near_rel("deserializer_free",
				    vec1,
				    deserializer.free_cov_matrix(L1));
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_cov_matrix(std_vec);

  stan::test::expect_near_rel("deserializer_free",
				    vec1,
				    std_uvec[0]);
  stan::test::expect_near_rel("deserializer_free",
				    vec2,
				    std_uvec[1]);

  EXPECT_THROW(deserializer.free_cov_matrix(L_bad), std::invalid_argument);
  EXPECT_THROW(deserializer.free_cov_matrix(std_vec_bad), std::invalid_argument);
}

// corr_matrix

TEST(deserializer_vector, corr_matrix_free) {
  int M = 3;
  int P = (M * (M - 1)) / 2;
  Eigen::VectorXd vec1 = Eigen::VectorXd::Random(P);
  Eigen::VectorXd vec2 = Eigen::VectorXd::Random(P);

  Eigen::MatrixXd L1 = stan::math::corr_matrix_constrain(vec1, M);
  Eigen::MatrixXd L2 = stan::math::corr_matrix_constrain(vec2, M);
  Eigen::MatrixXd L_bad(2, 3);
  
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::deserializer<double> deserializer(theta, theta_i);

  std::vector<Eigen::MatrixXd> std_vec = { L1, L2 },
    std_vec_bad = { L1, L_bad };

  stan::test::expect_near_rel("deserializer_free",
				    vec1,
				    deserializer.free_corr_matrix(L1));
  std::vector<Eigen::VectorXd> std_uvec = deserializer.free_corr_matrix(std_vec);

  stan::test::expect_near_rel("deserializer_free",
				    vec1,
				    std_uvec[0]);
  stan::test::expect_near_rel("deserializer_free",
				    vec2,
				    std_uvec[1]);

  EXPECT_THROW(deserializer.free_corr_matrix(L_bad), std::invalid_argument);
  EXPECT_THROW(deserializer.free_corr_matrix(std_vec_bad), std::invalid_argument);
}
