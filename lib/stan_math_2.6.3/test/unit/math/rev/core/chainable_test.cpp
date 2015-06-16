#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

namespace chainable_test {
  class mock_chainable : public stan::math::chainable {
  public:
    mock_chainable() {
      throw std::invalid_argument("hello");
    }
  };
}

TEST(AgradRevChainable,ctorThrow) {
  EXPECT_THROW(chainable_test::mock_chainable(), std::invalid_argument);
}
