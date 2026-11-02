#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/random_var_context.hpp>
#include <test/test-models/good/services/partial_init_dependent.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The missing parameters must take exactly the random unconstrained values
// of the first attempt, and the provided parameters must keep their
// constrained values.
class ServicesUtilInitializePartial : public testing::Test {
 public:
  ServicesUtilInitializePartial()
      : model(empty_context, 12345, &model_ss),
        rng(stan::services::util::create_rng(0, 1)) {}

  using values_t = std::vector<std::pair<std::string, double>>;

  // One variable of the init context, in var_context layout.
  struct entry {
    std::string name;
    std::vector<double> vals;
    std::vector<size_t> dims;
  };

  static stan::io::array_var_context make_context(
      const std::vector<entry>& entries) {
    std::vector<std::string> names;
    std::vector<double> vals;
    std::vector<std::vector<size_t>> dims;
    for (const auto& e : entries) {
      names.push_back(e.name);
      vals.insert(vals.end(), e.vals.begin(), e.vals.end());
      dims.push_back(e.dims);
    }
    return stan::io::array_var_context(names, vals, dims);
  }

  static stan::io::array_var_context make_context(const values_t& values) {
    std::vector<entry> entries;
    for (const auto& nv : values) {
      entries.push_back({nv.first, {nv.second}, {}});
    }
    return make_context(entries);
  }

  static std::string base_name(const std::string& name) {
    return name.substr(0, name.find_first_of(".:"));
  }

  // Runs initialize with scalar init values and checks the result.
  void check(const values_t& provided, int seed, double init_radius) {
    std::vector<entry> entries;
    for (const auto& nv : provided) {
      entries.push_back({nv.first, {nv.second}, {}});
    }
    check(entries, provided, seed, init_radius);
  }

  // Runs initialize with the given seed and checks the result.  expected
  // holds the constrained values of the provided parameters by flat name
  // (the names of constrained_param_names).
  void check(const std::vector<entry>& provided, const values_t& expected,
             int seed, double init_radius) {
    stan::io::array_var_context init_context = make_context(provided);
    rng = stan::services::util::create_rng(seed, 1);
    stan::rng_t rng_copy = rng;
    stan::io::random_var_context random_context(model, rng_copy, init_radius,
                                                init_radius == 0);
    const std::vector<double> u_random = random_context.get_unconstrained();

    std::vector<double> params = stan::services::util::initialize(
        model, init_context, rng, init_radius, false, logger, init);
    ASSERT_EQ(u_random.size(), params.size());
    EXPECT_EQ(0, logger.call_count_warn()) << "seed " << seed;

    std::set<std::string> provided_bases;
    for (const auto& e : provided) {
      provided_bases.insert(base_name(e.name));
    }
    std::vector<std::string> u_names;
    model.unconstrained_param_names(u_names, false, false);
    ASSERT_EQ(params.size(), u_names.size());
    for (size_t n = 0; n < params.size(); ++n) {
      if (provided_bases.count(base_name(u_names[n])) == 0) {
        EXPECT_EQ(u_random[n], params[n])
            << "seed " << seed << ", " << u_names[n];
      }
    }

    std::vector<std::string> c_names;
    model.constrained_param_names(c_names, false, false);
    std::vector<int> params_i;
    std::vector<double> vars;
    model.write_array(rng, params, params_i, vars, false, false);
    ASSERT_EQ(c_names.size(), vars.size());
    for (const auto& nv : expected) {
      size_t found = 0;
      for (size_t n = 0; n < c_names.size(); ++n) {
        if (c_names[n] == nv.first) {
          ++found;
          EXPECT_NEAR(nv.second, vars[n],
                      1e-12 * std::fmax(1.0, std::fabs(nv.second)))
              << "seed " << seed << ", " << nv.first;
        }
      }
      EXPECT_EQ(1, found) << nv.first;
    }
  }

  stan_model model;
  stan::io::empty_var_context empty_context;
  std::stringstream model_ss;
  stan::test::unit::instrumented_logger logger;
  stan::test::unit::instrumented_writer init;
  stan::rng_t rng;
};

TEST_F(ServicesUtilInitializePartial, small_multiplier) {
  // x, v and t.2 have multiplier=sigma: freeing a random constrained value
  // with sigma = 1e-20 gives an unconstrained value of order 1e20.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"sigma", 1e-20}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, provided_offset) {
  // x has offset=mu: freeing a random constrained value with mu = 5 shifts
  // x by (mu_random - 5) / sigma.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"mu", 5}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, radius_zero) {
  // With init_radius = 0 the missing parameters are 0 on the unconstrained
  // scale, also when their offset depends on a provided parameter.
  check({{"mu", 5}}, 0, 0);
}

TEST_F(ServicesUtilInitializePartial, provided_dependent_parameter) {
  // x is provided and depends on the missing mu and sigma.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"x", 3}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, chain_provided_a_c) {
  // b depends on the provided a, and the provided c depends on b.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"a", 1}, {"c", 4}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, chain_provided_b_d) {
  // The provided b depends on the missing a; the missing c depends on b;
  // the provided d depends on c.  d needs one correction pass.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"b", 2}, {"d", -3}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, chain_provided_a_c_e) {
  // Provided a, c and e alternate with the missing b and d.  c is correct
  // after one correction pass, and then e after a second one.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"a", 1}, {"c", 4}, {"e", -2}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, provided_vector) {
  // v has multiplier=sigma, and both are provided.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"sigma", {1e-20}, {}}, {"v", {0.5, -0.25}, {2}}},
          {{"sigma", 1e-20}, {"v.1", 0.5}, {"v.2", -0.25}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, provided_simplex) {
  for (int seed = 0; seed < 10; ++seed) {
    check({{"s", {0.2, 0.3, 0.5}, {3}}},
          {{"s.1", 0.2}, {"s.2", 0.3}, {"s.3", 0.5}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, provided_array_of_tuples) {
  // The init context stores at slot by slot; write_array stores it element
  // by element.  at.2 has multiplier=sigma with a missing sigma.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"at.1", {0.3, -0.7}, {2}}, {"at.2", {1.5, 2.5}, {2}}},
          {{"at.1:1", 0.3}, {"at.1:2", 1.5}, {"at.2:1", -0.7}, {"at.2:2", 2.5}},
          seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, offset_from_array_of_tuples) {
  // The provided z has offset=at[2].1, where at is a missing array of
  // tuples.  write_array stores at element by element, but a var_context
  // stores it slot by slot.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"z", 2}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, bound_on_provided_value_is_rejected) {
  // Known limitation: y has lower=lb.  With init_radius = 0 the random
  // constrained value of y is 1, below the provided lb = 10, so the only
  // attempt is rejected.  The valid start y = 11 is not found.
  stan::io::array_var_context init_context = make_context({{"lb", 10}});
  EXPECT_THROW(stan::services::util::initialize(model, init_context, rng, 0,
                                                false, logger, init),
               std::domain_error);
}
