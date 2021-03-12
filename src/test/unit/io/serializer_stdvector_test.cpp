#include <stan/io/serializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>


// vector

TEST(serializer_stdvec_vector, read) {
  std::vector<double> theta;
  std::vector<Eigen::VectorXd> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    Eigen::VectorXd x_vec(10);
    for (size_t i = 0; i < 10U; ++i) {
      theta.push_back(static_cast<double>(i));
      x_vec.coeffRef(i) = -static_cast<double>(i);
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
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}


TEST(serializer_stdvec_vector, complex_read) {
  std::vector<double> theta;
  using complex_vec = Eigen::Matrix<std::complex<double>, -1, 1>;
  std::vector<complex_vec> x_stdvec;
  for (size_t i = 0; i < 40U; ++i) {
    theta.push_back(static_cast<double>(i));
  }
  for (size_t i = 0; i < 2U; ++i) {
    complex_vec x(10);
    for (size_t j = 0; j < 10U; ++j) {
      x.coeffRef(j) = std::complex<double>(-static_cast<double>(j), -static_cast<double>(j + 1));
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
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}
// row vector

TEST(serializer_stdvec_rowvector, read) {
  std::vector<double> theta;
  std::vector<Eigen::RowVectorXd> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    Eigen::RowVectorXd x_vec(10);
    for (size_t i = 0; i < 10U; ++i) {
      theta.push_back(static_cast<double>(i));
      x_vec.coeffRef(i) = -static_cast<double>(i);
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
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}


TEST(serializer_stdvec_rowvector, complex_read) {
  std::vector<double> theta;
  using complex_vec = Eigen::Matrix<std::complex<double>, 1, -1>;
  std::vector<complex_vec> x_stdvec;
  for (size_t i = 0; i < 40U; ++i) {
    theta.push_back(static_cast<double>(i));
  }
  for (size_t i = 0; i < 2U; ++i) {
    complex_vec x(10);
    for (size_t j = 0; j < 10U; ++j) {
      x.coeffRef(j) = std::complex<double>(-static_cast<double>(j), -static_cast<double>(j + 1));
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
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// matrix


TEST(serializer_stdvec_matrix, read) {
  std::vector<double> theta;
  std::vector<Eigen::MatrixXd> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    Eigen::MatrixXd x_vec(4, 4);
    for (size_t i = 0; i < 16U; ++i) {
      theta.push_back(static_cast<double>(i));
      x_vec.coeffRef(i) = -static_cast<double>(i);
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
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}


TEST(serializer_stdvec_matrix, complex_read) {
  std::vector<double> theta;
  using complex_mat = Eigen::Matrix<std::complex<double>, -1, -1>;
  std::vector<complex_mat> x_stdvec;
  for (size_t i = 0; i < 64U; ++i) {
    theta.push_back(static_cast<double>(i));
  }
  for (size_t i = 0; i < 2U; ++i) {
    complex_mat x(4, 4);
    for (size_t j = 0; j < 16U; ++j) {
      x.coeffRef(j) = std::complex<double>(-static_cast<double>(j), -static_cast<double>(j + 1));
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
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}


TEST(serializer_stdvec_stdvector, read) {
  std::vector<double> theta;
  std::vector<std::vector<double>> x_stdvec;
  for (size_t i = 0; i < 2U; ++i) {
    std::vector<double> x_vec;
    for (size_t i = 0; i < 10U; ++i) {
      theta.push_back(static_cast<double>(i));
      x_vec.push_back(-static_cast<double>(i));
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
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}


TEST(serializer_stdvec_stdvector, complex_read) {
  std::vector<double> theta;
  using complex_vec = std::vector<std::complex<double>>;
  std::vector<complex_vec> x_stdvec;
  for (size_t i = 0; i < 40U; ++i) {
    theta.push_back(static_cast<double>(i));
  }
  for (size_t i = 0; i < 2U; ++i) {
    complex_vec x;
    for (size_t j = 0; j < 10U; ++j) {
      x.push_back(std::complex<double>(-static_cast<double>(j), -static_cast<double>(j + 1)));
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
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}
