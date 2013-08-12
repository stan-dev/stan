#include <stan/diff/rev/chainable.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

namespace chainable_test {
  class mock_chainable : public stan::diff::chainable {
  public:
    mock_chainable() {
      throw std::invalid_argument("hello");
    }
  };
}

TEST(AgradRevChainable,ctorThrow) {
  EXPECT_THROW(chainable_test::mock_chainable(), std::invalid_argument);
}
