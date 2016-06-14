#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/mcmc/hmc/nuts_classic/base_nuts_classic.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>
#include <boost/random/additive_combine.hpp>
#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

namespace stan {
  namespace mcmc {

    class mock_nuts_classic:
      public base_nuts_classic<mock_model, mock_hamiltonian,
                               mock_integrator, rng_t> {
    public:
      mock_nuts_classic(const mock_model &m, rng_t& rng):
        base_nuts_classic<mock_model, mock_hamiltonian,
                          mock_integrator, rng_t>(m, rng)
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

    class divergent_nuts_classic:
      public base_nuts_classic<mock_model, divergent_hamiltonian,
                               expl_leapfrog, rng_t> {
    public:
      divergent_nuts_classic(const mock_model &m, rng_t& rng):
        base_nuts_classic<mock_model, divergent_hamiltonian,
                          expl_leapfrog, rng_t>(m, rng)
      { }

    private:
      bool compute_criterion(ps_point& start,
                             ps_point& finish,
                             Eigen::VectorXd& rho) { return false; }
    };

  }
}

TEST(McmcNutsBaseNutsClassic, set_max_depth) {

  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::mock_nuts_classic sampler(model, base_rng);

  int old_max_depth = 1;
  sampler.set_max_depth(old_max_depth);
  EXPECT_EQ(old_max_depth, sampler.get_max_depth());

  sampler.set_max_depth(-1);
  EXPECT_EQ(old_max_depth, sampler.get_max_depth());
}


TEST(McmcNutsBaseNuts, set_max_delta) {
  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::mock_nuts_classic sampler(model, base_rng);

  double old_max_delta = 10;
  sampler.set_max_delta(old_max_delta);
  EXPECT_EQ(old_max_delta, sampler.get_max_delta());
}

TEST(McmcNutsBaseNutsClassic, build_tree) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  Eigen::VectorXd rho = Eigen::VectorXd::Zero(model_size);

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::ps_point z_propose(model_size);

  stan::mcmc::nuts_util util;
  util.log_u = -1;
  util.H0 = -0.1;
  util.sign = 1;
  util.n_tree = 0;
  util.sum_prob = 0;

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::mock_nuts_classic sampler(model, base_rng);

  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);


  int n_valid = sampler.build_tree(3, rho, &z_init, z_propose, util,
                                   writer, error_writer);

  EXPECT_EQ(8, n_valid);

  EXPECT_EQ(8, util.n_tree);
  EXPECT_FLOAT_EQ(std::exp(util.H0) * util.n_tree, util.sum_prob);

  EXPECT_EQ(init_momentum * util.n_tree, rho(0));

  EXPECT_EQ(init_momentum, z_init.q(0));
  EXPECT_EQ(init_momentum, z_init.p(0));

  EXPECT_EQ(8 * init_momentum, sampler.z().q(0));
  EXPECT_EQ(init_momentum, sampler.z().p(0));

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error_stream.str());
}

TEST(McmcNutsBaseNutsClassic, slice_criterion) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  Eigen::VectorXd rho = Eigen::VectorXd::Zero(model_size);

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::ps_point z_propose(model_size);

  stan::mcmc::nuts_util util;
  util.log_u = 0;
  util.H0 = 0;
  util.sign = 1;
  util.n_tree = 0;
  util.sum_prob = 0;

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::divergent_nuts_classic sampler(model, base_rng);

  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);


  int n_valid = 0;

  sampler.z().V = -750;
  n_valid = sampler.build_tree(0, rho, &z_init, z_propose, util,
                               writer, error_writer);

  EXPECT_EQ(1, n_valid);
  EXPECT_EQ(0, sampler.divergent_);

  sampler.z().V = -250;
  n_valid = sampler.build_tree(0, rho, &z_init, z_propose, util,
                               writer, error_writer);

  EXPECT_EQ(0, n_valid);
  EXPECT_EQ(0, sampler.divergent_);

  sampler.z().V = 750;
  n_valid = sampler.build_tree(0, rho, &z_init, z_propose, util,
                               writer, error_writer);

  EXPECT_EQ(0, n_valid);
  EXPECT_EQ(1, sampler.divergent_);

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error_stream.str());
}
