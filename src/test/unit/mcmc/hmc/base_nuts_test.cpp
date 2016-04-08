#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>
#include <boost/random/additive_combine.hpp>
#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

namespace stan {
  namespace mcmc {

    class mock_nuts: public base_nuts<mock_model,
                                      mock_hamiltonian,
                                      mock_integrator,
                                      rng_t> {

    public:
      mock_nuts(mock_model &m, rng_t& rng)
        : base_nuts<mock_model,mock_hamiltonian,mock_integrator,rng_t>(m, rng)
      { }

    private:

      bool compute_criterion(ps_point& start,
                             ps_point& finish,
                             Eigen::VectorXd& rho) { return true; }

    };

    // Mock Hamiltonian
    template <typename M, typename BaseRNG>
    class divergent_hamiltonian
      : public base_hamiltonian<M, ps_point, BaseRNG> {
    public:
      divergent_hamiltonian(M& m)
        : base_hamiltonian<M, ps_point, BaseRNG>(m) {}

      double T(ps_point& z) { return 0; }

      double tau(ps_point& z) { return T(z); }
      double phi(ps_point& z) { return this->V(z); }

      const Eigen::VectorXd dtau_dq(ps_point& z) {
        return Eigen::VectorXd::Zero(this->model_.num_params_r());
      }

      const Eigen::VectorXd dtau_dp(ps_point& z) {
        return Eigen::VectorXd::Zero(this->model_.num_params_r());
      }

      const Eigen::VectorXd dphi_dq(ps_point& z) {
        return Eigen::VectorXd::Zero(this->model_.num_params_r());
      }

      void init(ps_point& z,
                stan::interface_callbacks::writer::base_writer& writer) {
        z.V = 0;
      }

      void sample_p(ps_point& z, BaseRNG& rng) {};

      void update(ps_point& z,
                  stan::interface_callbacks::writer::base_writer& writer) {
        z.V += 500;
      }

    };

    class divergent_nuts: public base_nuts<mock_model,
                                           divergent_hamiltonian,
                                           expl_leapfrog,
                                           rng_t> {

    public:

      divergent_nuts(mock_model &m, rng_t& rng)
        : base_nuts<mock_model, divergent_hamiltonian, expl_leapfrog,rng_t>(m, rng)
      { }

    private:

      bool compute_criterion(ps_point& start,
                             ps_point& finish,
                             Eigen::VectorXd& rho) { return false; }

    };

  }
}

TEST(McmcBaseNuts, set_max_depth) {

  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::mock_nuts sampler(model, base_rng);

  int old_max_depth = 1;
  sampler.set_max_depth(old_max_depth);
  EXPECT_EQ(old_max_depth, sampler.get_max_depth());

  sampler.set_max_depth(-1);
  EXPECT_EQ(old_max_depth, sampler.get_max_depth());
}


TEST(McmcBaseNuts, set_max_delta) {
  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::mock_nuts sampler(model, base_rng);

  double old_max_delta = 10;
  sampler.set_max_delta(old_max_delta);
  EXPECT_EQ(old_max_delta, sampler.get_max_delta());
}

TEST(McmcBaseNuts, build_tree) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::ps_point z_propose(model_size);

  Eigen::VectorXd rho = z_init.p;
  double sum_weight = 0;

  double H0 = -0.1;
  int n_leapfrog = 0;
  double sum_metro_prob = 0;

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::mock_nuts sampler(model, base_rng);

  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);

  bool valid_subtree = sampler.build_tree(3, rho, z_propose,
                                          H0, 1, n_leapfrog, sum_weight,
                                          sum_metro_prob, writer);

  EXPECT_TRUE(valid_subtree);

  EXPECT_EQ(init_momentum * (n_leapfrog + 1), rho(0));

  EXPECT_EQ(8 * init_momentum, sampler.z().q(0));
  EXPECT_EQ(init_momentum, sampler.z().p(0));

  EXPECT_EQ(8, n_leapfrog);
  EXPECT_FLOAT_EQ(std::exp(H0) * n_leapfrog, sum_weight);
  EXPECT_FLOAT_EQ(std::exp(H0) * n_leapfrog, sum_metro_prob);

  EXPECT_EQ("", output.str());
}

TEST(McmcBaseNuts, divergence_test) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::ps_point z_propose(model_size);

  Eigen::VectorXd rho = z_init.p;
  double sum_weight = 0;

  double H0 = -0.1;
  int n_leapfrog = 0;
  double sum_metro_prob = 0;

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::divergent_nuts sampler(model, base_rng);

  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);

  bool valid_subtree = 0;

  sampler.z().V = -750;
  valid_subtree = sampler.build_tree(0, rho, z_propose,
                                     H0, 1, n_leapfrog, sum_weight,
                                     sum_metro_prob, writer);
  EXPECT_TRUE(valid_subtree);
  EXPECT_EQ(0, sampler.divergent_);

  sampler.z().V = -250;
  valid_subtree = sampler.build_tree(0, rho, z_propose,
                                     H0, 1, n_leapfrog, sum_weight,
                                     sum_metro_prob, writer);

  EXPECT_TRUE(valid_subtree);
  EXPECT_EQ(0, sampler.divergent_);

  sampler.z().V = 750;
  valid_subtree = sampler.build_tree(0, rho, z_propose,
                                     H0, 1, n_leapfrog, sum_weight,
                                     sum_metro_prob, writer);

  EXPECT_FALSE(valid_subtree);
  EXPECT_EQ(1, sampler.divergent_);

  EXPECT_EQ("", output.str());
}

TEST(McmcBaseNuts, transition) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::mock_nuts sampler(model, base_rng);

  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, writer);

  EXPECT_EQ(31.5, s.cont_params()(0));
  EXPECT_EQ(0, s.log_prob());
  EXPECT_EQ(1, s.accept_stat());
  EXPECT_EQ("", output.str());
}
