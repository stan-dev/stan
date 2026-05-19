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

TEST(psPoint, get_param_names_appends_to_existing_names) {
  ps_point point(2);
  std::vector<std::string> model_names{"alpha", "beta"};
  std::vector<std::string> names{"lp__"};

  point.get_param_names(model_names, names);

  std::vector<std::string> expected{"lp__",   "alpha",   "beta",  "p_alpha",
                                    "p_beta", "g_alpha", "g_beta"};
  EXPECT_EQ(expected, names);
}

TEST(psPoint, get_params_appends_to_existing_values) {
  ps_point point(2);
  point.q << 1.0, 2.0;
  point.p << 3.0, 4.0;
  point.g << 5.0, 6.0;
  std::vector<double> values{-1.0};

  point.get_params(values);

  std::vector<double> expected{-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  EXPECT_EQ(expected, values);
}

}  // namespace mcmc
}  // namespace stan
