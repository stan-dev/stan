#include <gtest/gtest.h>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <test/unit/util.hpp>

namespace stan {
namespace mcmc {

TEST(psPoint, write_metric_streams) {
  stan::test::capture_std_streams();

  ps_point point(2);
  std::stringstream out;
  stan::callbacks::stream_writer writer(out);
  EXPECT_NO_THROW(point.write_metric(writer));
  EXPECT_EQ("", out.str());

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}

}  // namespace mcmc
}  // namespace stan
