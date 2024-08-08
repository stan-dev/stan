#include <stan/io/serializer.hpp>
#include <stan/io/deserializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

// lb

namespace stan {
namespace test {
template <typename Ret, typename DeserializeRead, typename DeserializeFree,
          typename... Args, typename... Sizes>
void deserializer_test_impl(DeserializeRead&& deserialize_read,
                            DeserializeFree&& deserialize_free,
                            const std::tuple<Sizes...>& sizes, Args&&... args) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  Eigen::VectorXd theta3 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer1(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = stan::math::apply(deserialize_read, sizes, deserializer1, args..., lp);
  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write(vec_ref);
  // Read a serialized constrained variable and unconstrain it
  // This is the code that is being tested
  stan::io::deserializer<double> deserializer2(theta2, theta_i);
  Ret uvec_ref
      = stan::math::apply(deserialize_free, sizes, deserializer2, args...);
  //  deserialize_free(deserializer2, sizes...);

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

template <typename Ret, template <typename> class Deserializer,
          typename... Args, typename... Sizes>
void deserializer_test(const std::tuple<Sizes...>& sizes, Args&&... args) {
  deserializer_test_impl<Ret>(Deserializer<Ret>::read(),
                              Deserializer<Ret>::free(), sizes, args...);
}

}  // namespace test
}  // namespace stan

template <typename Ret>
struct LbConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_lb<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_lb<Ret>(args...);
    };
  }
};

TEST(deserializer_vector, read_free_lb) {
  using stan::test::deserializer_test;
  deserializer_test<double, LbConstrain>(std::make_tuple(), 0.5);
  deserializer_test<Eigen::VectorXd, LbConstrain>(std::make_tuple(4), 0.5);
  deserializer_test<std::vector<Eigen::VectorXd>, LbConstrain>(
      std::make_tuple(2, 4), 0.5);
  deserializer_test<std::vector<std::vector<Eigen::VectorXd>>, LbConstrain>(
      std::make_tuple(3, 2, 4), 0.5);
}

// ub
template <typename Ret>
struct UbConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_ub<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_ub<Ret>(args...);
    };
  }
};

TEST(deserializer_vector, read_free_ub) {
  using stan::test::deserializer_test;
  deserializer_test<double, UbConstrain>(std::make_tuple(), 0.5);
  deserializer_test<Eigen::VectorXd, UbConstrain>(std::make_tuple(4), 0.5);
  deserializer_test<std::vector<Eigen::VectorXd>, UbConstrain>(
      std::make_tuple(2, 4), 0.5);
  deserializer_test<std::vector<std::vector<Eigen::VectorXd>>, UbConstrain>(
      std::make_tuple(3, 2, 4), 0.5);
}

// lub
template <typename Ret>
struct LubConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_lub<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_lub<Ret>(args...);
    };
  }
};

TEST(deserializer_vector, read_free_lub) {
  using stan::test::deserializer_test;
  deserializer_test<double, LubConstrain>(std::make_tuple(), 0.2, 0.5);
  deserializer_test<Eigen::VectorXd, LubConstrain>(std::make_tuple(4), 0.2,
                                                   0.5);
  deserializer_test<std::vector<Eigen::VectorXd>, LubConstrain>(
      std::make_tuple(2, 4), 0.2, 0.5);
  deserializer_test<std::vector<std::vector<Eigen::VectorXd>>, LubConstrain>(
      std::make_tuple(3, 2, 4), 0.2, 0.5);
}

// offset multiplier
template <typename Ret>
struct OffsetMultConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_offset_multiplier<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_offset_multiplier<Ret>(args...);
    };
  }
};

TEST(deserializer_vector, read_free_offset_multiplier) {
  using stan::test::deserializer_test;
  deserializer_test<double, OffsetMultConstrain>(std::make_tuple(), 0.2, 0.5);
  deserializer_test<Eigen::VectorXd, OffsetMultConstrain>(std::make_tuple(4),
                                                          0.2, 0.5);
  deserializer_test<std::vector<Eigen::VectorXd>, OffsetMultConstrain>(
      std::make_tuple(2, 4), 0.2, 0.5);
  deserializer_test<std::vector<std::vector<Eigen::VectorXd>>,
                    OffsetMultConstrain>(std::make_tuple(3, 2, 4), 0.2, 0.5);
}

// unit vector
template <typename Ret>
struct UnitVecConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_unit_vector<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_unit_vector<Ret>(args...);
    };
  }
};

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
  using stan::test::deserializer_test;
  read_free_unit_vector_test<Eigen::VectorXd>(4);
  read_free_unit_vector_test<std::vector<Eigen::VectorXd>>(2, 4);
  read_free_unit_vector_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2,
                                                                        4);
}

// simplex
template <typename Ret>
struct SimplexConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_simplex<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_simplex<Ret>(args...);
    };
  }
};

TEST(deserializer_vector, read_free_simplex) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::VectorXd, SimplexConstrain>(std::make_tuple(4));
  deserializer_test<std::vector<Eigen::VectorXd>, SimplexConstrain>(
      std::make_tuple(2, 4));
  deserializer_test<std::vector<std::vector<Eigen::VectorXd>>,
                    SimplexConstrain>(std::make_tuple(3, 2, 4));
}

// sum_to_zero
template <typename Ret>
struct SumToZeroConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_sum_to_zero<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_sum_to_zero<Ret>(args...);
    };
  }
};

TEST(deserializer_vector, read_free_sum_to_zero) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::VectorXd, SumToZeroConstrain>(std::make_tuple(4));
  deserializer_test<std::vector<Eigen::VectorXd>, SumToZeroConstrain>(
      std::make_tuple(2, 4));
  deserializer_test<std::vector<std::vector<Eigen::VectorXd>>,
                    SumToZeroConstrain>(std::make_tuple(3, 2, 4));
}

// ordered
template <typename Ret>
struct OrderedConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_ordered<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_ordered<Ret>(args...);
    };
  }
};

TEST(deserializer_vector, read_free_ordered) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::VectorXd, OrderedConstrain>(std::make_tuple(4));
  deserializer_test<std::vector<Eigen::VectorXd>, OrderedConstrain>(
      std::make_tuple(2, 4));
  deserializer_test<std::vector<std::vector<Eigen::VectorXd>>,
                    OrderedConstrain>(std::make_tuple(3, 2, 4));
}

// positive_ordered
template <typename Ret>
struct PositiveOrderedConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_positive_ordered<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_positive_ordered<Ret>(args...);
    };
  }
};

TEST(deserializer_vector, read_free_positive_ordered) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::VectorXd, PositiveOrderedConstrain>(
      std::make_tuple(4));
  deserializer_test<std::vector<Eigen::VectorXd>, PositiveOrderedConstrain>(
      std::make_tuple(2, 4));
  deserializer_test<std::vector<std::vector<Eigen::VectorXd>>,
                    PositiveOrderedConstrain>(std::make_tuple(3, 2, 4));
}

// cholesky_factor_cov
template <typename Ret>
struct CholFacCovConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_cholesky_factor_cov<Ret, false>(
          args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_cholesky_factor_cov<Ret>(args...);
    };
  }
};
TEST(deserializer_vector, read_free_cholesky_factor_cov) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::MatrixXd, CholFacCovConstrain>(
      std::make_tuple(4, 3));
  deserializer_test<std::vector<Eigen::MatrixXd>, CholFacCovConstrain>(
      std::make_tuple(2, 4, 3));
  deserializer_test<std::vector<std::vector<Eigen::MatrixXd>>,
                    CholFacCovConstrain>(std::make_tuple(3, 2, 4, 3));

  deserializer_test<Eigen::MatrixXd, CholFacCovConstrain>(
      std::make_tuple(2, 2));
  deserializer_test<std::vector<Eigen::MatrixXd>, CholFacCovConstrain>(
      std::make_tuple(2, 2, 2));
  deserializer_test<std::vector<std::vector<Eigen::MatrixXd>>,
                    CholFacCovConstrain>(std::make_tuple(3, 2, 2, 2));
}

// cholesky_factor_corr
template <typename Ret>
struct CholFacCorrConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_cholesky_factor_corr<Ret, false>(
          args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_cholesky_factor_corr<Ret>(args...);
    };
  }
};
TEST(deserializer_vector, read_free_cholesky_factor_corr) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::MatrixXd, CholFacCorrConstrain>(std::make_tuple(2));
  deserializer_test<std::vector<Eigen::MatrixXd>, CholFacCorrConstrain>(
      std::make_tuple(2, 2));
  deserializer_test<std::vector<std::vector<Eigen::MatrixXd>>,
                    CholFacCorrConstrain>(std::make_tuple(3, 2, 2));
}

// cov_matrix
template <typename Ret>
struct CovMatConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_cov_matrix<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_cov_matrix<Ret>(args...);
    };
  }
};
TEST(deserializer_vector, read_free_cov_matrix) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::MatrixXd, CovMatConstrain>(std::make_tuple(2));
  deserializer_test<std::vector<Eigen::MatrixXd>, CovMatConstrain>(
      std::make_tuple(2, 2));
  deserializer_test<std::vector<std::vector<Eigen::MatrixXd>>, CovMatConstrain>(
      std::make_tuple(3, 2, 2));
}

// corr_matrix
template <typename Ret>
struct CorrMatConstrain {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_constrain_corr_matrix<Ret, false>(args...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto&&... args) {
      return deserializer.read_free_corr_matrix<Ret>(args...);
    };
  }
};
TEST(deserializer_vector, read_free_corr_matrix) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::MatrixXd, CorrMatConstrain>(std::make_tuple(2));
  deserializer_test<std::vector<Eigen::MatrixXd>, CorrMatConstrain>(
      std::make_tuple(2, 2));
  deserializer_test<std::vector<std::vector<Eigen::MatrixXd>>,
                    CorrMatConstrain>(std::make_tuple(3, 2, 2));
}

// stochastic_column
template <typename Ret>
struct StochasticCol {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto& lp,
              auto... sizes) {
      return deserializer.read_constrain_stochastic_column<Ret, false>(
          lp, sizes...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto... sizes) {
      return deserializer.read_free_stochastic_column<Ret>(sizes...);
    };
  }
};
TEST(deserializer_vector, read_stochastic_column_matrix) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::MatrixXd, StochasticCol>(std::make_tuple(3, 3));
  deserializer_test<std::vector<Eigen::MatrixXd>, StochasticCol>(
      std::make_tuple(2, 3, 3));
  deserializer_test<std::vector<std::vector<Eigen::MatrixXd>>, StochasticCol>(
      std::make_tuple(3, 2, 3, 3));
}

template <typename Ret>
struct StochasticRow {
  static auto read() {
    return [](stan::io::deserializer<double>& deserializer, auto& lp,
              auto... sizes) {
      return deserializer.read_constrain_stochastic_row<Ret, false>(lp,
                                                                    sizes...);
    };
  }
  static auto free() {
    return [](stan::io::deserializer<double>& deserializer, auto... sizes) {
      return deserializer.read_free_stochastic_row<Ret>(sizes...);
    };
  }
};
TEST(deserializer_vector, read_stochastic_row_matrix) {
  using stan::test::deserializer_test;
  deserializer_test<Eigen::MatrixXd, StochasticRow>(std::make_tuple(3, 3));
  deserializer_test<std::vector<Eigen::MatrixXd>, StochasticRow>(
      std::make_tuple(2, 3, 3));
  deserializer_test<std::vector<std::vector<Eigen::MatrixXd>>, StochasticRow>(
      std::make_tuple(3, 2, 3, 3));
}
