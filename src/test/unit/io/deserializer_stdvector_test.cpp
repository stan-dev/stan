#include <stan/io/deserializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

template <typename T, typename... Sizes>
void test_std_vector_deserializer_read(Sizes... sizes) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));

  stan::io::deserializer<double> deserializer1(theta, theta_i);
  stan::io::deserializer<double> deserializer2(theta, theta_i);

  auto y = deserializer1.read<std::vector<T>>(4, sizes...);
  for (size_t i = 0; i < 4; ++i) {
    stan::test::expect_near_rel("test_std_vector_deserializer", y[i],
                                deserializer2.read<T>(sizes...));
  }
}

TEST(deserializer_array, read_arrays) {
  test_std_vector_deserializer_read<double>();
  test_std_vector_deserializer_read<std::complex<double>>();
  test_std_vector_deserializer_read<std::vector<double>>(3);
  test_std_vector_deserializer_read<Eigen::VectorXd>(2);
  test_std_vector_deserializer_read<
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>>(2);
  test_std_vector_deserializer_read<std::vector<Eigen::VectorXd>>(3, 2);
  test_std_vector_deserializer_read<Eigen::RowVectorXd>(2);
  test_std_vector_deserializer_read<
      Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic>>(2);
  test_std_vector_deserializer_read<std::vector<Eigen::RowVectorXd>>(3, 2);
  test_std_vector_deserializer_read<Eigen::MatrixXd>(2, 2);
  test_std_vector_deserializer_read<
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>(2,
                                                                           2);
  test_std_vector_deserializer_read<std::vector<Eigen::MatrixXd>>(2, 3, 2);
}

// unit vector

TEST(deserializer_array, unit_vector) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));

  stan::io::deserializer<double> deserializer1(theta, theta_i);
  stan::io::deserializer<double> deserializer2(theta, theta_i);

  // no jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y
        = deserializer1
              .read_constrain_unit_vector<std::vector<Eigen::VectorXd>, false>(
                  lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_unit_vector<Eigen::VectorXd, false>(
              lp_ref, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y
        = deserializer1
              .read_constrain_unit_vector<std::vector<Eigen::VectorXd>, true>(
                  lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_unit_vector<Eigen::VectorXd, true>(
              lp_ref, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }
}

// simplex

TEST(deserializer_array, simplex) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));

  stan::io::deserializer<double> deserializer1(theta, theta_i);
  stan::io::deserializer<double> deserializer2(theta, theta_i);

  // no jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1
                 .read_constrain_simplex<std::vector<Eigen::VectorXd>, false>(
                     lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_simplex<Eigen::VectorXd, false>(lp_ref,
                                                                       3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1
                 .read_constrain_simplex<std::vector<Eigen::VectorXd>, true>(
                     lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_simplex<Eigen::VectorXd, true>(lp_ref,
                                                                      3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }
}

// ordered

TEST(deserializer_array, ordered) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));

  stan::io::deserializer<double> deserializer1(theta, theta_i);
  stan::io::deserializer<double> deserializer2(theta, theta_i);

  // no jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1
                 .read_constrain_ordered<std::vector<Eigen::VectorXd>, false>(
                     lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_ordered<Eigen::VectorXd, false>(lp_ref,
                                                                       3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1
                 .read_constrain_ordered<std::vector<Eigen::VectorXd>, true>(
                     lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_ordered<Eigen::VectorXd, true>(lp_ref,
                                                                      3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }
}

// positive ordered

TEST(deserializer_array, positive_ordered) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));

  stan::io::deserializer<double> deserializer1(theta, theta_i);
  stan::io::deserializer<double> deserializer2(theta, theta_i);

  // no jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1.read_constrain_positive_ordered<
        std::vector<Eigen::VectorXd>, false>(lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_positive_ordered<Eigen::VectorXd, false>(
              lp_ref, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1.read_constrain_positive_ordered<
        std::vector<Eigen::VectorXd>, true>(lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_positive_ordered<Eigen::VectorXd, true>(
              lp_ref, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }
}

// chol_cov

TEST(deserializer_array, chol_cov) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 200U; ++i)
    theta.push_back(static_cast<double>(i));

  stan::io::deserializer<double> deserializer1(theta, theta_i);
  stan::io::deserializer<double> deserializer2(theta, theta_i);

  // no jac, square
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1.read_constrain_cholesky_factor_cov<
        std::vector<Eigen::MatrixXd>, false>(lp, 4, 3, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2
              .read_constrain_cholesky_factor_cov<Eigen::MatrixXd, false>(
                  lp_ref, 3, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // jac, square
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1.read_constrain_cholesky_factor_cov<
        std::vector<Eigen::MatrixXd>, true>(lp, 4, 3, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2
              .read_constrain_cholesky_factor_cov<Eigen::MatrixXd, true>(lp_ref,
                                                                         3, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // no jac, non-square
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1.read_constrain_cholesky_factor_cov<
        std::vector<Eigen::MatrixXd>, false>(lp, 4, 3, 2);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2
              .read_constrain_cholesky_factor_cov<Eigen::MatrixXd, false>(
                  lp_ref, 3, 2));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // jac, non-square
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1.read_constrain_cholesky_factor_cov<
        std::vector<Eigen::MatrixXd>, true>(lp, 4, 3, 2);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2
              .read_constrain_cholesky_factor_cov<Eigen::MatrixXd, true>(lp_ref,
                                                                         3, 2));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }
}

// chol_corr

TEST(deserializer_array, chol_corr) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));

  stan::io::deserializer<double> deserializer1(theta, theta_i);
  stan::io::deserializer<double> deserializer2(theta, theta_i);

  // no jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1.read_constrain_cholesky_factor_corr<
        std::vector<Eigen::MatrixXd>, false>(lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2
              .read_constrain_cholesky_factor_corr<Eigen::MatrixXd, false>(
                  lp_ref, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1.read_constrain_cholesky_factor_corr<
        std::vector<Eigen::MatrixXd>, true>(lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2
              .read_constrain_cholesky_factor_corr<Eigen::MatrixXd, true>(
                  lp_ref, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }
}

// cov_matrix

TEST(deserializer_array, cov_matrix) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));

  stan::io::deserializer<double> deserializer1(theta, theta_i);
  stan::io::deserializer<double> deserializer2(theta, theta_i);

  // no jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y
        = deserializer1
              .read_constrain_cov_matrix<std::vector<Eigen::MatrixXd>, false>(
                  lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_cov_matrix<Eigen::MatrixXd, false>(
              lp_ref, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y = deserializer1
                 .read_constrain_cov_matrix<std::vector<Eigen::MatrixXd>, true>(
                     lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_cov_matrix<Eigen::MatrixXd, true>(lp_ref,
                                                                         3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }
}

// corr_matrix

TEST(deserializer_array, corr_matrix) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));

  stan::io::deserializer<double> deserializer1(theta, theta_i);
  stan::io::deserializer<double> deserializer2(theta, theta_i);

  // no jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y
        = deserializer1
              .read_constrain_corr_matrix<std::vector<Eigen::MatrixXd>, false>(
                  lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_corr_matrix<Eigen::MatrixXd, false>(
              lp_ref, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }

  // jac
  {
    double lp_ref = 0.0;
    double lp = 0.0;
    auto y
        = deserializer1
              .read_constrain_corr_matrix<std::vector<Eigen::MatrixXd>, true>(
                  lp, 4, 3);
    for (size_t i = 0; i < 4; ++i) {
      stan::test::expect_near_rel(
          "test_std_vector_deserializer", y[i],
          deserializer2.read_constrain_corr_matrix<Eigen::MatrixXd, true>(
              lp_ref, 3));
    }
    EXPECT_FLOAT_EQ(lp_ref, lp);
  }
}
