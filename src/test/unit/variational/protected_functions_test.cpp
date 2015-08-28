#include <stan/variational/advi.hpp>

#include <test/unit/util.hpp>
#include <gtest/gtest.h>

namespace stan {
  namespace variational {

    class mock_advi: public advi {

    public:
      mock_advi() : advi() {};

    };
  }
}

TEST(advi_test, asdf) {

  stan::variational::mock_advi test_advi;
  double wtf = 0.0;
  std::cout << wtf;

}
