#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>
#include <vector>
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
      mock_nuts(const mock_model &m, rng_t& rng)
        : base_nuts<mock_model,mock_hamiltonian,mock_integrator,rng_t>(m, rng)
      { }
    };

    class rho_inspector_mock_nuts: public base_nuts<mock_model,
                                                    mock_hamiltonian,
                                                    mock_integrator,
                                                    rng_t> {

    public:
      std::vector<double> rho_values;
      rho_inspector_mock_nuts(const mock_model &m, rng_t& rng)
        : base_nuts<mock_model,mock_hamiltonian,mock_integrator,rng_t>(m, rng)
      { }

      bool compute_criterion(Eigen::VectorXd& p_sharp_minus,
                             Eigen::VectorXd& p_sharp_plus,
                             Eigen::VectorXd& rho) {
        rho_values.push_back(rho(0));
        return true;
      }
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

      double dG_dt(ps_point& z,
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

    class divergent_nuts: public base_nuts<mock_model,
                                           divergent_hamiltonian,
                                           expl_leapfrog,
                                           rng_t> {

    public:

      divergent_nuts(const mock_model &m, rng_t& rng)
        : base_nuts<mock_model, divergent_hamiltonian, expl_leapfrog,rng_t>(m, rng)
      { }
    };

  }
}

TEST(McmcNutsBaseNuts, set_max_depth_test) {

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


TEST(McmcNutsBaseNuts, set_max_delta_test) {
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

TEST(McmcNutsBaseNuts, build_tree_test) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::ps_point z_propose(model_size);

  Eigen::VectorXd p_sharp_left = Eigen::VectorXd::Zero(model_size);
  Eigen::VectorXd p_sharp_right = Eigen::VectorXd::Zero(model_size);
  Eigen::VectorXd rho = z_init.p;
  double log_sum_weight = -std::numeric_limits<double>::infinity();

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
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);


  bool valid_subtree = sampler.build_tree(3, z_propose,
                                          p_sharp_left, p_sharp_right, rho,
                                          H0, 1, n_leapfrog, log_sum_weight,
                                          sum_metro_prob, writer, error_writer);

  EXPECT_TRUE(valid_subtree);

  EXPECT_EQ(init_momentum * (n_leapfrog + 1), rho(0));
  EXPECT_EQ(1, p_sharp_left(0));
  EXPECT_EQ(1, p_sharp_right(0));

  EXPECT_EQ(8 * init_momentum, sampler.z().q(0));
  EXPECT_EQ(init_momentum, sampler.z().p(0));

  EXPECT_EQ(8, n_leapfrog);
  EXPECT_FLOAT_EQ(H0 + std::log(n_leapfrog), log_sum_weight);
  EXPECT_FLOAT_EQ(std::exp(H0) * n_leapfrog, sum_metro_prob);

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error_stream.str());
}

TEST(McmcNutsBaseNuts, rho_aggregation_test) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::ps_point z_propose(model_size);

  Eigen::VectorXd p_sharp_left = Eigen::VectorXd::Zero(model_size);
  Eigen::VectorXd p_sharp_right = Eigen::VectorXd::Zero(model_size);
  Eigen::VectorXd rho = z_init.p;
  double log_sum_weight = -std::numeric_limits<double>::infinity();

  double H0 = -0.1;
  int n_leapfrog = 0;
  double sum_metro_prob = 0;

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::rho_inspector_mock_nuts sampler(model, base_rng);

  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;

  std::stringstream output;
  stan::interface_callbacks::writer::stream_writer writer(output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);


  sampler.build_tree(3, z_propose,
                     p_sharp_left, p_sharp_right, rho,
                     H0, 1, n_leapfrog, log_sum_weight,
                     sum_metro_prob, writer, error_writer);

  EXPECT_EQ(7, sampler.rho_values.size());
  EXPECT_EQ(2 * init_momentum, sampler.rho_values.at(0));
  EXPECT_EQ(2 * init_momentum, sampler.rho_values.at(1));
  EXPECT_EQ(4 * init_momentum, sampler.rho_values.at(2));
  EXPECT_EQ(2 * init_momentum, sampler.rho_values.at(3));
  EXPECT_EQ(2 * init_momentum, sampler.rho_values.at(4));
  EXPECT_EQ(4 * init_momentum, sampler.rho_values.at(5));
  EXPECT_EQ(8 * init_momentum, sampler.rho_values.at(6));
}

TEST(McmcNutsBaseNuts, divergence_test) {

  rng_t base_rng(0);

  int model_size = 1;
  double init_momentum = 1.5;

  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;

  stan::mcmc::ps_point z_propose(model_size);

  Eigen::VectorXd p_sharp_left = Eigen::VectorXd::Zero(model_size);
  Eigen::VectorXd p_sharp_right = Eigen::VectorXd::Zero(model_size);
  Eigen::VectorXd rho = z_init.p;
  double log_sum_weight = -std::numeric_limits<double>::infinity();

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
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);


  bool valid_subtree = 0;

  sampler.z().V = -750;
  valid_subtree = sampler.build_tree(0, z_propose,
                                     p_sharp_left, p_sharp_right, rho,
                                     H0, 1, n_leapfrog, log_sum_weight,
                                     sum_metro_prob,
                                     writer, error_writer);
  EXPECT_TRUE(valid_subtree);
  EXPECT_FALSE(sampler.divergent_);

  sampler.z().V = -250;
  valid_subtree = sampler.build_tree(0, z_propose,
                                     p_sharp_left, p_sharp_right, rho,
                                     H0, 1, n_leapfrog, log_sum_weight,
                                     sum_metro_prob,
                                     writer, error_writer);

  EXPECT_TRUE(valid_subtree);
  EXPECT_FALSE(sampler.divergent_);

  sampler.z().V = 750;
  valid_subtree = sampler.build_tree(0, z_propose,
                                     p_sharp_left, p_sharp_right, rho,
                                     H0, 1, n_leapfrog, log_sum_weight,
                                     sum_metro_prob,
                                     writer, error_writer);

  EXPECT_FALSE(valid_subtree);
  EXPECT_TRUE(sampler.divergent_);

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error_stream.str());
}

TEST(McmcNutsBaseNuts, transition) {

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

  std::stringstream output_stream;
  stan::interface_callbacks::writer::stream_writer writer(output_stream);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  // Transition will expand trajectory until max_depth is hit
  stan::mcmc::sample s = sampler.transition(init_sample, writer, error_writer);

  EXPECT_EQ(sampler.get_max_depth(), sampler.depth_);
  EXPECT_EQ((2 << (sampler.get_max_depth() - 1)) - 1, sampler.n_leapfrog_);
  EXPECT_FALSE(sampler.divergent_);

  EXPECT_EQ(21 * init_momentum, s.cont_params()(0));
  EXPECT_EQ(0, s.log_prob());
  EXPECT_EQ(1, s.accept_stat());
  EXPECT_EQ("", output_stream.str());
  EXPECT_EQ("", error_stream.str());
}
