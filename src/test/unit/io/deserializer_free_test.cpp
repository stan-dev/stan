#include <stan/io/serializer.hpp>
#include <stan/io/deserializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

// lb

template <typename Ret, typename... Sizes>
void read_free_lb_test(Sizes... sizes) {
  double lb = 0.5;
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer1.read_constrain_lb<Ret, false>(lb, lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_lb<Ret>(lb, sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_lb) {
  read_free_lb_test<double>();
  read_free_lb_test<Eigen::VectorXd>(4);
  read_free_lb_test<std::vector<Eigen::VectorXd>>(2, 4);
  read_free_lb_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

// ub

template <typename Ret, typename... Sizes>
void read_free_ub_test(Sizes... sizes) {
  double ub = 0.5;
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer1.read_constrain_ub<Ret, false>(ub, lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_ub<Ret>(ub, sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_ub) {
  read_free_ub_test<double>();
  read_free_ub_test<Eigen::VectorXd>(4);
  read_free_ub_test<std::vector<Eigen::VectorXd>>(2, 4);
  read_free_ub_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

// lub

template <typename Ret, typename... Sizes>
void read_free_lub_test(Sizes... sizes) {
  double lb = 0.2;
  double ub = 0.5;
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer1.read_constrain_lub<Ret, false>(lb, ub, lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_lub<Ret>(lb, ub, sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_lub) {
  read_free_lub_test<double>();
  read_free_lub_test<Eigen::VectorXd>(4);
  read_free_lub_test<std::vector<Eigen::VectorXd>>(2, 4);
  read_free_lub_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

// offset multiplier

template <typename Ret, typename... Sizes>
void read_free_offset_multiplier_test(Sizes... sizes) {
  double offset = 0.5;
  double multiplier = 0.35;
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer1.read_constrain_offset_multiplier<Ret, false>(
      offset, multiplier, lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_offset_multiplier<Ret>(
      offset, multiplier, sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_offset_multiplier) {
  read_free_offset_multiplier_test<double>();
  read_free_offset_multiplier_test<Eigen::VectorXd>(4);
  read_free_offset_multiplier_test<std::vector<Eigen::VectorXd>>(2, 4);
  read_free_offset_multiplier_test<std::vector<std::vector<Eigen::VectorXd>>>(
      3, 2, 4);
}

// unit vector

template <typename Ret, typename... Sizes>
void read_free_unit_vector_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer1.read_constrain_unit_vector<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_unit_vector<Ret>(sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  // For unit vector, it's not actually doing a change of variables so we check
  // theta2 equals theta3 (freeing doesn't actually get the unconstrained
  // variable back).
  size_t used2 = theta2.size() - deserializer2.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used2, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta2.segment(0, used2),
                              theta3.segment(0, used2));
}

TEST(deserializer_vector, read_free_unit_vector) {
  read_free_unit_vector_test<Eigen::VectorXd>(4);
  read_free_unit_vector_test<std::vector<Eigen::VectorXd>>(2, 4);
  read_free_unit_vector_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2,
                                                                        4);
}

// simplex

template <typename Ret, typename... Sizes>
void read_free_simplex_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer1.read_constrain_simplex<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_simplex<Ret>(sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_simplex) {
  read_free_simplex_test<Eigen::VectorXd>(4);
  read_free_simplex_test<std::vector<Eigen::VectorXd>>(2, 4);
  read_free_simplex_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

// ordered

template <typename Ret, typename... Sizes>
void read_free_ordered_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer1.read_constrain_ordered<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_ordered<Ret>(sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_ordered) {
  read_free_ordered_test<Eigen::VectorXd>(4);
  read_free_ordered_test<std::vector<Eigen::VectorXd>>(2, 4);
  read_free_ordered_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

// positive_ordered

template <typename Ret, typename... Sizes>
void read_free_positive_ordered_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer1.read_constrain_positive_ordered<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_positive_ordered<Ret>(sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_positive_ordered) {
  read_free_positive_ordered_test<Eigen::VectorXd>(4);
  read_free_positive_ordered_test<std::vector<Eigen::VectorXd>>(2, 4);
  read_free_positive_ordered_test<std::vector<std::vector<Eigen::VectorXd>>>(
      3, 2, 4);
}

// cholesky_factor_cov

template <typename Ret, typename... Sizes>
void read_free_cholesky_factor_cov_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer1.read_constrain_cholesky_factor_cov<Ret, false>(
      lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_cholesky_factor_cov<Ret>(sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_cholesky_factor_cov) {
  read_free_cholesky_factor_cov_test<Eigen::MatrixXd>(4, 3);
  read_free_cholesky_factor_cov_test<std::vector<Eigen::MatrixXd>>(2, 4, 3);
  read_free_cholesky_factor_cov_test<std::vector<std::vector<Eigen::MatrixXd>>>(
      3, 2, 4, 3);

  read_free_cholesky_factor_cov_test<Eigen::MatrixXd>(2, 2);
  read_free_cholesky_factor_cov_test<std::vector<Eigen::MatrixXd>>(2, 2, 2);
  read_free_cholesky_factor_cov_test<std::vector<std::vector<Eigen::MatrixXd>>>(
      3, 2, 2, 2);
}

// cholesky_factor_corr

template <typename Ret, typename... Sizes>
void read_free_cholesky_factor_corr_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer1.read_constrain_cholesky_factor_corr<Ret, false>(
      lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_cholesky_factor_corr<Ret>(sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_cholesky_factor_corr) {
  read_free_cholesky_factor_corr_test<Eigen::MatrixXd>(2);
  read_free_cholesky_factor_corr_test<std::vector<Eigen::MatrixXd>>(2, 2);
  read_free_cholesky_factor_corr_test<
      std::vector<std::vector<Eigen::MatrixXd>>>(3, 2, 2);
}

// cov_matrix

template <typename Ret, typename... Sizes>
void read_free_cov_matrix_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer1.read_constrain_cov_matrix<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_cov_matrix<Ret>(sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_cov_matrix) {
  read_free_cov_matrix_test<Eigen::MatrixXd>(2);
  read_free_cov_matrix_test<std::vector<Eigen::MatrixXd>>(2, 2);
  read_free_cov_matrix_test<std::vector<std::vector<Eigen::MatrixXd>>>(3, 2, 2);
}

// corr_matrix

template <typename Ret, typename... Sizes>
void read_free_corr_matrix_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer1.read_constrain_corr_matrix<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);

  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref = deserializer2.read_free_corr_matrix<Ret>(sizes...);

  // Serialize the unconstrained variable
  stan::io::serializer<double> serializer2(theta3);
  serializer2.write(uvec_ref);

  size_t used1 = theta1.size() - deserializer1.available();
  size_t used3 = theta3.size() - serializer2.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used3);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta3.segment(0, used1));
}

TEST(deserializer_vector, read_free_corr_matrix) {
  read_free_corr_matrix_test<Eigen::MatrixXd>(2);
  read_free_corr_matrix_test<std::vector<Eigen::MatrixXd>>(2, 2);
  read_free_corr_matrix_test<std::vector<std::vector<Eigen::MatrixXd>>>(3, 2,
                                                                        2);
}
