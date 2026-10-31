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

  static stan::io::array_var_context make_context(const values_t& values) {
    std::vector<std::string> names;
    std::vector<double> vals;
    std::vector<std::vector<size_t>> dims;
    for (const auto& nv : values) {
      names.push_back(nv.first);
      vals.push_back(nv.second);
      dims.push_back({});
    }
    return stan::io::array_var_context(names, vals, dims);
  }

  static std::string base_name(const std::string& name) {
    return name.substr(0, name.find_first_of(".:"));
  }

  // Runs initialize with the given seed and checks the result.
  void check(const values_t& provided, int seed, double init_radius) {
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
    for (const auto& nv : provided) {
      provided_bases.insert(base_name(nv.first));
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
    for (const auto& nv : provided) {
      for (size_t n = 0; n < c_names.size(); ++n) {
        if (c_names[n] == nv.first) {
          EXPECT_NEAR(nv.second, vars[n], 1e-12 * std::fabs(nv.second))
              << "seed " << seed << ", " << nv.first;
        }
      }
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
  // the provided d depends on c.  This needs more than one pass.
  for (int seed = 0; seed < 10; ++seed) {
    check({{"b", 2}, {"d", -3}}, seed, 2);
  }
}

TEST_F(ServicesUtilInitializePartial, bound_on_provided_value_is_rejected) {
  // Documented limitation: y has lower=lb.  With init_radius = 0 the random
  // constrained value of y is 1, below the provided lb = 10, so the only
  // attempt is rejected.
  stan::io::array_var_context init_context = make_context({{"lb", 10}});
  EXPECT_THROW(stan::services::util::initialize(model, init_context, rng, 0,
                                                false, logger, init),
               std::domain_error);
}
