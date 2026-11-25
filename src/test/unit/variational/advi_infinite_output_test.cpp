#include <test/test-models/good/variational/univariate_no_constraint.hpp>
#include <stan/variational/advi.hpp>
#include <stan/services/util/create_rng.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

typedef stan::variational::advi<stan_model, stan::variational::normal_meanfield,
                                stan::rng_t>
    advi_meanfield_t;

// Infinite values cannot be read back from CSV output (Windows writes them as
// "1.#INF"), so they must be coerced to the largest finite double.
TEST(AdviInfiniteOutput, InfinityCoercedToMaxDouble) {
  const double max_val = std::numeric_limits<double>::max();
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> values{0, 1.5, inf, -inf};
  advi_meanfield_t::coerce_infinite_to_max(values);
  EXPECT_EQ(0, values[0]);
  EXPECT_EQ(1.5, values[1]);
  EXPECT_EQ(max_val, values[2]);
  EXPECT_EQ(-max_val, values[3]);
}
