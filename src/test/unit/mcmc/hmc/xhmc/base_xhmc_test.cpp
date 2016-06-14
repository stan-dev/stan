#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/mcmc/hmc/xhmc/base_xhmc.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>
#include <boost/random/additive_combine.hpp>
#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

namespace stan {
  namespace mcmc {

    class mock_xhmc: public base_xhmc<mock_model,
                                      mock_hamiltonian,
                                      mock_integrator,
                                      rng_t> {

    public:
      mock_xhmc(const mock_model &m, rng_t& rng)
        : base_xhmc<mock_model,mock_hamiltonian,mock_integrator,rng_t>(m, rng)
      { }
    };

    // Mock Hamiltonian
    template <typename M, typename BaseRNG>
    class divergent_hamiltonian
      : public base_hamiltonian<M, ps_point, BaseRNG> {
    public:
      divergent_hamiltonian(const M& m)
        : base_hamiltonian<M, ps_point, BaseRNG>(m) {}

      double T(ps_point& z) { return 0; }

      double tau(ps_point& z) { return T(z); }
      double phi(ps_point& z) { return this->V(z); }

      double dG_dt(
        ps_point& z,
        interface_callbacks::writer::base_writer& info_writer,
        interface_callbacks::writer::base_writer& error_writer) {
        return 2;
      }

      Eigen::VectorXd dtau_dq(
        ps_point& z,
        interface_callbacks::writer::base_writer& info_writer,
        interface_callbacks::writer::base_writer& error_writer) {
        return Eigen::VectorXd::Zero(this->model_.num_params_r());
      }

      Eigen::VectorXd dtau_dp(ps_point& z) {
        return Eigen::VectorXd::Zero(this->model_.num_params_r());
      }

      Eigen::VectorXd dphi_dq(
        ps_point& z,
        interface_callbacks::writer::base_writer& info_writer,
        interface_callbacks::writer::base_writer& error_writer) {
        return Eigen::VectorXd::Zero(this->model_.num_params_r());
      }

      void init(ps_point& z,
                interface_callbacks::writer::base_writer& info_writer,
                interface_callbacks::writer::base_writer& error_writer) {
        z.V = 0;
      }

      void sample_p(ps_point& z, BaseRNG& rng) {};

      void update_potential_gradient(
        ps_point& z,
        interface_callbacks::writer::base_writer& info_writer,
        interface_callbacks::writer::base_writer& error_writer) {
        z.V += 500;
      }

    };

    class divergent_xhmc: public base_xhmc<mock_model,
                                           divergent_hamiltonian,
                                           expl_leapfrog,
                                           rng_t> {
    public:

      divergent_xhmc(const mock_model &m, rng_t& rng)
        : base_xhmc<mock_model, divergent_hamiltonian, expl_leapfrog,rng_t>(m, rng)
      { }
    };

  }
}

TEST(McmcXHMCBaseXHMC, set_max_depth) {

  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::mock_xhmc sampler(model, base_rng);

  int old_max_depth = 1;
  sampler.set_max_depth(old_max_depth);
  EXPECT_EQ(old_max_depth, sampler.get_max_depth());

  sampler.set_max_depth(-1);
  EXPECT_EQ(old_max_depth, sampler.get_max_depth());
}


TEST(McmcXHMCBaseXHMC, set_max_deltaH) {
  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::mock_xhmc sampler(model, base_rng);

  double old_max_deltaH = 10;
  sampler.set_max_deltaH(old_max_deltaH);
  EXPECT_EQ(old_max_deltaH, sampler.get_max_deltaH());
}

TEST(McmcXHMCBaseXHMC, set_x_delta) {
  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::mock_xhmc sampler(model, base_rng);

  double old_x_delta = 1;
  sampler.set_x_delta(old_x_delta);
  EXPECT_EQ(old_x_delta, sampler.get_x_delta());
}

TEST(McmcXHMCBaseXHMC, build_tree) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::ps_point z_propose(model_size);

  double ave = 0;
  double log_sum_weight = -std::numeric_limits<double>::infinity();

  double H0 = -0.1;
  int n_leapfrog = 0;
  double sum_metro_prob = 0;

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::mock_xhmc sampler(model, base_rng);

  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);


  bool valid_subtree = sampler.build_tree(3, z_propose,
                                          ave, log_sum_weight,
                                          H0, 1, n_leapfrog,
                                          sum_metro_prob,
                                          writer, error_writer);

  EXPECT_TRUE(valid_subtree);

  EXPECT_EQ(8 * init_momentum, sampler.z().q(0));
  EXPECT_EQ(init_momentum, sampler.z().p(0));

  EXPECT_EQ(8, n_leapfrog);
  EXPECT_FLOAT_EQ(2, ave);
  EXPECT_FLOAT_EQ(H0  + std::log(n_leapfrog), log_sum_weight);
  EXPECT_FLOAT_EQ(std::exp(H0) * n_leapfrog, sum_metro_prob);

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error_stream.str());
}

TEST(McmcXHMCBaseXHMC, divergence_test) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::ps_point z_propose(model_size);

  double ave = 0;
  double log_sum_weight = -std::numeric_limits<double>::infinity();

  double H0 = -0.1;
  int n_leapfrog = 0;
  double sum_metro_prob = 0;

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::divergent_xhmc sampler(model, base_rng);

  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);


  bool valid_subtree = 0;

  sampler.z().V = -750;
  valid_subtree = sampler.build_tree(0, z_propose,
                                     ave, log_sum_weight,
                                     H0, 1, n_leapfrog,
                                     sum_metro_prob,
                                     writer, error_writer);
  EXPECT_TRUE(valid_subtree);
  EXPECT_EQ(0, sampler.divergent_);

  sampler.z().V = -250;
  valid_subtree = sampler.build_tree(0, z_propose,
                                     ave, log_sum_weight,
                                     H0, 1, n_leapfrog,
                                     sum_metro_prob,
                                     writer, error_writer);

  EXPECT_TRUE(valid_subtree);
  EXPECT_EQ(0, sampler.divergent_);

  sampler.z().V = 750;
  valid_subtree = sampler.build_tree(0, z_propose,
                                     ave, log_sum_weight,
                                     H0, 1, n_leapfrog,
                                     sum_metro_prob,
                                     writer, error_writer);

  EXPECT_FALSE(valid_subtree);
  EXPECT_EQ(1, sampler.divergent_);

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error_stream.str());
}

TEST(McmcXHMCBaseXHMC, transition) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::mock_xhmc sampler(model, base_rng);

  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;

  std::stringstream output_stream;
  stan::interface_callbacks::writer::stream_writer writer(output_stream);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, writer, error_writer);

  EXPECT_EQ(31.5, s.cont_params()(0));
  EXPECT_EQ(0, s.log_prob());
  EXPECT_EQ(1, s.accept_stat());
  EXPECT_EQ("", output_stream.str());
  EXPECT_EQ("", error_stream.str());
}
