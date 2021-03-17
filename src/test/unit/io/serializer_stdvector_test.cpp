#include <stan/io/serializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

// vector

TEST(serializer_stdvec_vector, write) {
  std::vector<double> theta(100);
  std::vector<Eigen::VectorXd> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    Eigen::VectorXd x_vec(10);
    for (size_t j = 0; j < 10U; ++j) {
      x_vec.coeffRef(j) = -static_cast<double>(i * 10 + j);
    }
    x_stdvec.push_back(x_vec);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x_stdvec);
  int sentinal = 0;
  for (size_t i = 0; i < 2U; ++i) {
    for (size_t j = 0; j < 10U; ++j) {
      EXPECT_FLOAT_EQ(theta[sentinal], x_stdvec[i][j]);
      ++sentinal;
    }
  }
}

TEST(serializer_stdvec_vector, complex_write) {
  std::vector<double> theta(100);
  using complex_vec = Eigen::Matrix<std::complex<double>, -1, 1>;
  std::vector<complex_vec> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    complex_vec x(10);
    for (size_t j = 0; j < 10U; ++j) {
      x.coeffRef(j)
          = std::complex<double>(-static_cast<double>(i * 20 + j),
                                 -static_cast<double>(i * 20 + j + 1));
    }
    x_stdvec.push_back(x);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x_stdvec);
  Eigen::Index sentinal = 0;
  for (size_t i = 0; i < 2U; ++i) {
    for (size_t j = 0; j < 10U; ++j) {
      EXPECT_FLOAT_EQ(theta[sentinal], x_stdvec[i][j].real())
          << "For idx: " << sentinal << " i: " << i << "j: " << j << "\n";
      EXPECT_FLOAT_EQ(theta[sentinal + 1], x_stdvec[i][j].imag())
          << "For idx: " << sentinal << " i: " << i << "j: " << j << "\n";
      sentinal += 2;
    }
  }
}
// row vector

TEST(serializer_stdvec_rowvector, write) {
  std::vector<double> theta(100);
  std::vector<Eigen::RowVectorXd> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    Eigen::RowVectorXd x_vec(10);
    for (size_t j = 0; j < 10U; ++j) {
      x_vec.coeffRef(j) = -static_cast<double>(i * 10 + j);
    }
    x_stdvec.push_back(x_vec);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x_stdvec);
  int sentinal = 0;
  for (size_t i = 0; i < 2U; ++i) {
    for (size_t j = 0; j < 10U; ++j) {
      EXPECT_FLOAT_EQ(theta[sentinal], x_stdvec[i][j]);
      ++sentinal;
    }
  }
}

TEST(serializer_stdvec_rowvector, complex_write) {
  std::vector<double> theta(100);
  using complex_vec = Eigen::Matrix<std::complex<double>, 1, -1>;
  std::vector<complex_vec> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    complex_vec x(10);
    for (size_t j = 0; j < 10U; ++j) {
      x.coeffRef(j)
          = std::complex<double>(-static_cast<double>(i * 20 + j),
                                 -static_cast<double>(i * 20 + j + 1));
    }
    x_stdvec.push_back(x);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x_stdvec);
  Eigen::Index sentinal = 0;
  for (size_t i = 0; i < 2U; ++i) {
    for (size_t j = 0; j < 10U; ++j) {
      EXPECT_FLOAT_EQ(theta[sentinal], x_stdvec[i][j].real())
          << "For idx: " << sentinal << " i: " << i << "j: " << j << "\n";
      EXPECT_FLOAT_EQ(theta[sentinal + 1], x_stdvec[i][j].imag())
          << "For idx: " << sentinal << " i: " << i << "j: " << j << "\n";
      sentinal += 2;
    }
  }
}

// matrix

TEST(serializer_stdvec_matrix, write) {
  std::vector<double> theta(100);
  std::vector<Eigen::MatrixXd> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    Eigen::MatrixXd x_vec(4, 4);
    for (size_t j = 0; j < 16U; ++j) {
      x_vec.coeffRef(j) = -static_cast<double>(i * 16 + j);
    }
    x_stdvec.push_back(x_vec);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x_stdvec);
  int sentinal = 0;
  for (size_t i = 0; i < 2U; ++i) {
    for (size_t j = 0; j < 16U; ++j) {
      EXPECT_FLOAT_EQ(theta[sentinal], x_stdvec[i](j));
      ++sentinal;
    }
  }
}

TEST(serializer_stdvec_matrix, complex_write) {
  std::vector<double> theta(100);
  using complex_mat = Eigen::Matrix<std::complex<double>, -1, -1>;
  std::vector<complex_mat> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    complex_mat x(4, 4);
    for (size_t j = 0; j < 16U; ++j) {
      x.coeffRef(j)
          = std::complex<double>(-static_cast<double>(i * 32 + j),
                                 -static_cast<double>(i * 32 + j + 1));
    }
    x_stdvec.push_back(x);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x_stdvec);
  Eigen::Index sentinal = 0;
  for (size_t i = 0; i < 2U; ++i) {
    for (size_t j = 0; j < 16U; ++j) {
      EXPECT_FLOAT_EQ(theta[sentinal], x_stdvec[i](j).real())
          << "For idx: " << sentinal << " i: " << i << "j: " << j << "\n";
      EXPECT_FLOAT_EQ(theta[sentinal + 1], x_stdvec[i](j).imag())
          << "For idx: " << sentinal << " i: " << i << "j: " << j << "\n";
      sentinal += 2;
    }
  }
}

TEST(serializer_stdvec_stdvector, write) {
  std::vector<double> theta(100);
  std::vector<std::vector<double>> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    std::vector<double> x_vec;
    for (size_t j = 0; j < 10U; ++j) {
      x_vec.push_back(-static_cast<double>(i * 10 + j));
    }
    x_stdvec.push_back(x_vec);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x_stdvec);
  int sentinal = 0;
  for (size_t i = 0; i < 2U; ++i) {
    for (size_t j = 0; j < 10U; ++j) {
      EXPECT_FLOAT_EQ(theta[sentinal], x_stdvec[i][j]);
      ++sentinal;
    }
  }
}

TEST(serializer_stdvec_stdvector, complex_write) {
  std::vector<double> theta(100);
  using complex_vec = std::vector<std::complex<double>>;
  std::vector<complex_vec> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    complex_vec x;
    for (size_t j = 0; j < 10U; ++j) {
      x.push_back(std::complex<double>(-static_cast<double>(i * 20 + j),
                                       -static_cast<double>(i * 20 + j + 1)));
    }
    x_stdvec.push_back(x);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x_stdvec);
  Eigen::Index sentinal = 0;
  for (size_t i = 0; i < 2U; ++i) {
    for (size_t j = 0; j < 10U; ++j) {
      EXPECT_FLOAT_EQ(theta[sentinal], x_stdvec[i][j].real())
          << "For idx: " << sentinal << " i: " << i << "j: " << j << "\n";
      EXPECT_FLOAT_EQ(theta[sentinal + 1], x_stdvec[i][j].imag())
          << "For idx: " << sentinal << " i: " << i << "j: " << j << "\n";
      sentinal += 2;
    }
  }
}
