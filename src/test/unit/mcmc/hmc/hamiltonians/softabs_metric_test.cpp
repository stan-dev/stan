#include <stan/io/dump.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_metric.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/interface_callbacks/writer/noop_writer.hpp>

#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <test/test-models/good/mcmc/hmc/hamiltonians/funnel.hpp>
#include <test/unit/util.hpp>

#include <boost/random/additive_combine.hpp>

#include <gtest/gtest.h>

#include <string>

typedef boost::ecuyer1988 rng_t;

TEST(McmcSoftAbs, sample_p) {
  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::softabs_metric<stan::mcmc::mock_model, rng_t> metric(model);
  stan::mcmc::softabs_point z(q.size());

  int n_samples = 1000;
  double m = 0;
  double m2 = 0;

  std::stringstream model_output, metric_output;
  stan::interface_callbacks::writer::stream_writer writer(metric_output);

  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  metric.update_metric(z, writer, error_writer);

  for (int i = 0; i < n_samples; ++i) {
    metric.sample_p(z, base_rng);
    double tau = metric.tau(z);

    double delta = tau - m;
    m += delta / static_cast<double>(i + 1);
    m2 += delta * (tau - m);
  }

  double var = m2 / (n_samples + 1.0);

  // Mean within 5sigma of expected value (d / 2)
  EXPECT_TRUE(std::fabs(m   - 0.5 * q.size()) < 5.0 * sqrt(var));

  // Variance within 10% of expected value (d / 2)
  EXPECT_TRUE(std::fabs(var - 0.5 * q.size()) < 0.1 * q.size());

  EXPECT_EQ("", metric_output.str());
}

TEST(McmcSoftAbs, gradients) {

  rng_t base_rng(0);

  Eigen::VectorXd q = Eigen::VectorXd::Ones(11);

  stan::mcmc::softabs_point z(q.size());
  z.q = q;
  z.p.setOnes();

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream model_output, metric_output;
  stan::interface_callbacks::writer::stream_writer writer(metric_output);

  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  funnel_model_namespace::funnel_model model(data_var_context, &model_output);

  stan::mcmc::softabs_metric<funnel_model_namespace::funnel_model, rng_t> metric(model);

  double epsilon = 1e-6;

  metric.init(z, writer, error_writer);
  Eigen::VectorXd g1 = metric.dtau_dq(z, writer, error_writer);

  for (int i = 0; i < z.q.size(); ++i) {

    double delta = 0;

    z.q(i) += epsilon;
    metric.init(z, writer, error_writer);
    delta += metric.tau(z);

    z.q(i) -= 2 * epsilon;
    metric.init(z, writer, error_writer);
    delta -= metric.tau(z);

    z.q(i) += epsilon;

    delta /= 2 * epsilon;

    EXPECT_NEAR(delta, g1(i), epsilon);
  }

  metric.init(z, writer, error_writer);
  Eigen::VectorXd g2 = metric.dtau_dp(z);

  for (int i = 0; i < z.q.size(); ++i) {

    double delta = 0;

    z.p(i) += epsilon;
    delta += metric.tau(z);

    z.p(i) -= 2 * epsilon;
    delta -= metric.tau(z);

    z.p(i) += epsilon;

    delta /= 2 * epsilon;

    EXPECT_NEAR(delta, g2(i), epsilon);
  }

  Eigen::VectorXd g3 = metric.dphi_dq(z, writer, error_writer);

  for (int i = 0; i < z.q.size(); ++i) {
    double delta = 0;

    z.q(i) += epsilon;
    metric.init(z, writer, error_writer);
    delta += metric.phi(z);

    z.q(i) -= 2 * epsilon;
    metric.init(z, writer, error_writer);
    delta -= metric.phi(z);

    z.q(i) += epsilon;

    delta /= 2 * epsilon;

    EXPECT_NEAR(delta, g3(i), epsilon);
  }

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", metric_output.str());
  EXPECT_EQ("", error_stream.str());
}

TEST(McmcSoftAbs, streams) {
  stan::test::capture_std_streams();
  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  stan::mcmc::mock_model model(q.size());

  // for use in Google Test macros below
  typedef stan::mcmc::softabs_metric<stan::mcmc::mock_model, rng_t> softabs;

  EXPECT_NO_THROW(softabs metric(model));

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
