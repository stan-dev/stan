#include <string>
#include <boost/random/additive_combine.hpp>
#include <stan/io/dump.hpp>
#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <test/test-models/good/mcmc/hmc/hamiltonians/funnel.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

TEST(McmcDiagEMetric, sample_p) {

  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::diag_e_metric<stan::mcmc::mock_model, rng_t> metric(model);
  stan::mcmc::diag_e_point z(q.size());

  int n_samples = 1000;
  double m = 0;
  double m2 = 0;

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
}

TEST(McmcDiagEMetric, gradients) {
  rng_t base_rng(0);

  Eigen::VectorXd q = Eigen::VectorXd::Ones(11);

  stan::mcmc::diag_e_point z(q.size());
  z.q = q;
  z.p.setOnes();

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream model_output;
  funnel_model_namespace::funnel_model model(data_var_context, &model_output);


  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  stan::mcmc::diag_e_metric<funnel_model_namespace::funnel_model, rng_t> metric(model);

  double epsilon = 1e-6;

  metric.init(z, logger);
  Eigen::VectorXd g1 = metric.dtau_dq(z, logger);

  for (int i = 0; i < z.q.size(); ++i) {

    double delta = 0;

    z.q(i) += epsilon;
    metric.update_potential(z, logger);
    delta += metric.tau(z);

    z.q(i) -= 2 * epsilon;
    metric.update_potential(z, logger);
    delta -= metric.tau(z);

    z.q(i) += epsilon;
    metric.update_potential(z, logger);

    delta /= 2 * epsilon;

    EXPECT_NEAR(delta, g1(i), epsilon);

  }

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

  Eigen::VectorXd g3 = metric.dphi_dq(z, logger);

  for (int i = 0; i < z.q.size(); ++i) {

    double delta = 0;

    z.q(i) += epsilon;
    metric.update_potential(z, logger);
    delta += metric.phi(z);

    z.q(i) -= 2 * epsilon;
    metric.update_potential(z, logger);
    delta -= metric.phi(z);

    z.q(i) += epsilon;
    metric.update_potential(z, logger);

    delta /= 2 * epsilon;

    EXPECT_NEAR(delta, g3(i), epsilon);

  }
  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcDiagEMetric, streams) {
  stan::test::capture_std_streams();

  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;


  stan::mcmc::mock_model model(q.size());

  // typedef to use within Google Test macros
  typedef stan::mcmc::diag_e_metric<stan::mcmc::mock_model, rng_t> diag_e;

  EXPECT_NO_THROW(diag_e metric(model));

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
