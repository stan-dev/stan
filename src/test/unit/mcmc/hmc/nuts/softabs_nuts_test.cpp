#include <stan/math/rev/mat.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/mcmc/hmc/nuts/softabs_nuts.hpp>
#include <boost/random/additive_combine.hpp>
#include <test/test-models/good/mcmc/hmc/common/gauss3D.hpp>
#include <stan/io/dump.hpp>
#include <fstream>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

TEST(McmcSoftAbsNuts, build_tree_test) {
  rng_t base_rng(4839294);

  stan::mcmc::softabs_point z_init(3);
  z_init.q(0) = 1;
  z_init.q(1) = -1;
  z_init.q(2) = 1;
  z_init.p(0) = -1;
  z_init.p(1) = 1;
  z_init.p(2) = -1;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  std::fstream empty_stream("", std::fstream::in);
  stan::io::dump data_var_context(empty_stream);
  gauss3D_model_namespace::gauss3D_model model(data_var_context);

  stan::mcmc::softabs_nuts<gauss3D_model_namespace::gauss3D_model, rng_t>
    sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::ps_point z_propose = z_init;

  Eigen::VectorXd p_sharp_left = Eigen::VectorXd::Zero(z_init.p.size());
  Eigen::VectorXd p_sharp_right = Eigen::VectorXd::Zero(z_init.p.size());
  Eigen::VectorXd rho = z_init.p;
  double log_sum_weight = -std::numeric_limits<double>::infinity();

  double H0 = -0.1;
  int n_leapfrog = 0;
  double sum_metro_prob = 0;

  bool valid_subtree = sampler.build_tree(3, z_propose,
                                          p_sharp_left, p_sharp_right, rho,
                                          H0, 1, n_leapfrog, log_sum_weight,
                                          sum_metro_prob, writer, error_writer);

  EXPECT_EQ(0.1, sampler.get_nominal_stepsize());

  EXPECT_TRUE(valid_subtree);

  EXPECT_FLOAT_EQ(-11.679803, rho(0));
  EXPECT_FLOAT_EQ(11.679803, rho(1));
  EXPECT_FLOAT_EQ(-11.679803, rho(2));

  EXPECT_FLOAT_EQ(0.20423166, sampler.z().q(0));
  EXPECT_FLOAT_EQ(-0.20423166, sampler.z().q(1));
  EXPECT_FLOAT_EQ(0.20423166, sampler.z().q(2));

  EXPECT_FLOAT_EQ(-1.5019561, sampler.z().p(0));
  EXPECT_FLOAT_EQ(1.5019561, sampler.z().p(1));
  EXPECT_FLOAT_EQ(-1.5019561, sampler.z().p(2));

  EXPECT_EQ(8, n_leapfrog);
  EXPECT_FLOAT_EQ(std::log(0.34310558), log_sum_weight);
  EXPECT_FLOAT_EQ(0.34310558, sum_metro_prob);

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error_stream.str());
}

TEST(McmcSoftAbsNuts, tree_boundary_test) {
  rng_t base_rng(4839294);

  stan::mcmc::softabs_point z_init(3);
  z_init.q(0) = 1;
  z_init.q(1) = -1;
  z_init.q(2) = 1;
  z_init.p(0) = -1;
  z_init.p(1) = 1;
  z_init.p(2) = -1;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  std::fstream empty_stream("", std::fstream::in);
  stan::io::dump data_var_context(empty_stream);

  typedef gauss3D_model_namespace::gauss3D_model model_t;
  model_t model(data_var_context);

  // Compute expected tree boundaries
  typedef stan::mcmc::softabs_metric<model_t, rng_t> metric_t;
  metric_t metric(model);

  stan::mcmc::impl_leapfrog<metric_t> softabs_integrator;
  double epsilon = 0.1;

  stan::mcmc::softabs_point z_test = z_init;
  metric.init(z_test, writer, error_writer);

  softabs_integrator.evolve(z_test, metric, epsilon, writer, error_writer);
  Eigen::VectorXd p_sharp_forward_1 = metric.dtau_dp(z_test);

  softabs_integrator.evolve(z_test, metric, epsilon, writer, error_writer);
  Eigen::VectorXd p_sharp_forward_2 = metric.dtau_dp(z_test);

  softabs_integrator.evolve(z_test, metric, epsilon, writer, error_writer);
  softabs_integrator.evolve(z_test, metric, epsilon, writer, error_writer);
  Eigen::VectorXd p_sharp_forward_3 = metric.dtau_dp(z_test);

  softabs_integrator.evolve(z_test, metric, epsilon, writer, error_writer);
  softabs_integrator.evolve(z_test, metric, epsilon, writer, error_writer);
  softabs_integrator.evolve(z_test, metric, epsilon, writer, error_writer);
  softabs_integrator.evolve(z_test, metric, epsilon, writer, error_writer);
  Eigen::VectorXd p_sharp_forward_4 = metric.dtau_dp(z_test);

  z_test = z_init;
  metric.init(z_test, writer, error_writer);

  softabs_integrator.evolve(z_test, metric, -epsilon, writer, error_writer);
  Eigen::VectorXd p_sharp_backward_1 = metric.dtau_dp(z_test);

  softabs_integrator.evolve(z_test, metric, -epsilon, writer, error_writer);
  Eigen::VectorXd p_sharp_backward_2 = metric.dtau_dp(z_test);

  softabs_integrator.evolve(z_test, metric, -epsilon, writer, error_writer);
  softabs_integrator.evolve(z_test, metric, -epsilon, writer, error_writer);
  Eigen::VectorXd p_sharp_backward_3 = metric.dtau_dp(z_test);

  softabs_integrator.evolve(z_test, metric, -epsilon, writer, error_writer);
  softabs_integrator.evolve(z_test, metric, -epsilon, writer, error_writer);
  softabs_integrator.evolve(z_test, metric, -epsilon, writer, error_writer);
  softabs_integrator.evolve(z_test, metric, -epsilon, writer, error_writer);
  Eigen::VectorXd p_sharp_backward_4 = metric.dtau_dp(z_test);

  // Check expected tree boundaries to those dynamically geneated by NUTS
  stan::mcmc::softabs_nuts<model_t, rng_t> sampler(model, base_rng);

  sampler.set_nominal_stepsize(epsilon);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::ps_point z_propose = z_init;

  Eigen::VectorXd p_sharp_left = Eigen::VectorXd::Zero(z_init.p.size());
  Eigen::VectorXd p_sharp_right = Eigen::VectorXd::Zero(z_init.p.size());
  Eigen::VectorXd rho = z_init.p;
  double log_sum_weight = -std::numeric_limits<double>::infinity();

  double H0 = -0.1;
  int n_leapfrog = 0;
  double sum_metro_prob = 0;

  // Depth 0 forward
  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.build_tree(0, z_propose,
                     p_sharp_left, p_sharp_right, rho,
                     H0, 1, n_leapfrog, log_sum_weight,
                     sum_metro_prob,
                     writer, error_writer);

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_forward_1(n), p_sharp_left(n));

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_forward_1(n), p_sharp_right(n));

  // Depth 1 forward
  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.build_tree(1, z_propose,
                     p_sharp_left, p_sharp_right, rho,
                     H0, 1, n_leapfrog, log_sum_weight,
                     sum_metro_prob,
                     writer, error_writer);

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_forward_1(n), p_sharp_left(n));

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_forward_2(n), p_sharp_right(n));

  // Depth 2 forward
  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.build_tree(2, z_propose,
                     p_sharp_left, p_sharp_right, rho,
                     H0, 1, n_leapfrog, log_sum_weight,
                     sum_metro_prob,
                     writer, error_writer);

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_forward_1(n), p_sharp_left(n));

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_forward_3(n), p_sharp_right(n));

  // Depth 3 forward
  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.build_tree(3, z_propose,
                     p_sharp_left, p_sharp_right, rho,
                     H0, 1, n_leapfrog, log_sum_weight,
                     sum_metro_prob,
                     writer, error_writer);

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_forward_1(n), p_sharp_left(n));

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_forward_4(n), p_sharp_right(n));

  // Depth 0 backward
  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.build_tree(0, z_propose,
                     p_sharp_left, p_sharp_right, rho,
                     H0, -1, n_leapfrog, log_sum_weight,
                     sum_metro_prob,
                     writer, error_writer);

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_backward_1(n), p_sharp_left(n));

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_backward_1(n), p_sharp_right(n));

  // Depth 1 backward
  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.build_tree(1, z_propose,
                     p_sharp_left, p_sharp_right, rho,
                     H0, -1, n_leapfrog, log_sum_weight,
                     sum_metro_prob,
                     writer, error_writer);

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_backward_1(n), p_sharp_left(n));

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_backward_2(n), p_sharp_right(n));

  // Depth 2 backward
  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.build_tree(2, z_propose,
                     p_sharp_left, p_sharp_right, rho,
                     H0, -1, n_leapfrog, log_sum_weight,
                     sum_metro_prob,
                     writer, error_writer);

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_backward_1(n), p_sharp_left(n));

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_backward_3(n), p_sharp_right(n));

  // Depth 3 backward
  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.build_tree(3, z_propose,
                     p_sharp_left, p_sharp_right, rho,
                     H0, -1, n_leapfrog, log_sum_weight,
                     sum_metro_prob,
                     writer, error_writer);

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_backward_1(n), p_sharp_left(n));

  for (int n = 0; n < rho.size(); ++n)
    EXPECT_FLOAT_EQ(p_sharp_backward_4(n), p_sharp_right(n));
}

TEST(McmcSoftAbsNuts, transition_test) {
  rng_t base_rng(4839294);

  stan::mcmc::softabs_point z_init(3);
  z_init.q(0) = 1;
  z_init.q(1) = -1;
  z_init.q(2) = 1;
  z_init.p(0) = -1;
  z_init.p(1) = 1;
  z_init.p(2) = -1;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  std::fstream empty_stream("", std::fstream::in);
  stan::io::dump data_var_context(empty_stream);
  gauss3D_model_namespace::gauss3D_model model(data_var_context);

  stan::mcmc::softabs_nuts<gauss3D_model_namespace::gauss3D_model, rng_t>
    sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(writer, error_writer);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, writer, error_writer);

  EXPECT_EQ(4, sampler.depth_);
  EXPECT_EQ((2 << 3) - 1, sampler.n_leapfrog_);
  EXPECT_FALSE(sampler.divergent_);

  EXPECT_FLOAT_EQ(1.9313564, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-0.86902142, s.cont_params()(1));
  EXPECT_FLOAT_EQ(1.6008, s.cont_params()(2));
  EXPECT_FLOAT_EQ(-3.5239484, s.log_prob());
  EXPECT_FLOAT_EQ(0.99709648, s.accept_stat());
  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error_stream.str());
}
