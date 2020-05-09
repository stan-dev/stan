#ifndef STAN_MCMC_HMC_NUTS_BASE_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_BASE_NUTS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace stan {
namespace mcmc {
/**
 * The No-U-Turn sampler (NUTS) with multinomial sampling
 */
template <class Model, template <class, class> class Hamiltonian,
          template <class> class Integrator, class BaseRNG>
class base_nuts : public base_hmc<Model, Hamiltonian, Integrator, BaseRNG> {
 public:
  base_nuts(const Model& model, BaseRNG& rng)
      : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng),
        depth_(0),
        max_depth_(5),
        max_deltaH_(1000),
        n_leapfrog_(0),
        divergent_(false),
        energy_(0) {}

  /**
   * specialized constructor for specified diag mass matrix
   */
  base_nuts(const Model& model, BaseRNG& rng, Eigen::VectorXd& inv_e_metric)
      : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng,
                                                          inv_e_metric),
        depth_(0),
        max_depth_(5),
        max_deltaH_(1000),
        n_leapfrog_(0),
        divergent_(false),
        energy_(0) {}

  /**
   * specialized constructor for specified dense mass matrix
   */
  base_nuts(const Model& model, BaseRNG& rng, Eigen::MatrixXd& inv_e_metric)
      : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng,
                                                          inv_e_metric),
        depth_(0),
        max_depth_(5),
        max_deltaH_(1000),
        n_leapfrog_(0),
        divergent_(false),
        energy_(0) {}

  ~base_nuts() {}

  void set_metric(const Eigen::MatrixXd& inv_e_metric) {
    this->z_.set_metric(inv_e_metric);
  }

  void set_metric(const Eigen::VectorXd& inv_e_metric) {
    this->z_.set_metric(inv_e_metric);
  }

  void set_max_depth(int d) {
    if (d > 0)
      max_depth_ = d;
  }

  void set_max_delta(double d) { max_deltaH_ = d; }

  int get_max_depth() { return this->max_depth_; }
  double get_max_delta() { return this->max_deltaH_; }

  sample transition(sample& init_sample, callbacks::logger& logger) {
    // Initialize the algorithm
    this->sample_stepsize();

    this->seed(init_sample.cont_params());

    this->hamiltonian_.sample_p(this->z_, this->rand_int_);
    this->hamiltonian_.init(this->z_, logger);

    ps_point z_fwd(this->z_);  // State at forward end of trajectory
    ps_point z_bck(z_fwd);     // State at backward end of trajectory

    ps_point z_sample(z_fwd);
    ps_point z_propose(z_fwd);

    // Momentum and sharp momentum at forward end of forward subtree
    Eigen::VectorXd p_fwd_fwd = this->z_.p;
    Eigen::VectorXd p_sharp_fwd_fwd = this->hamiltonian_.dtau_dp(this->z_);

    // Momentum and sharp momentum at backward end of forward subtree
    Eigen::VectorXd p_fwd_bck = this->z_.p;
    Eigen::VectorXd p_sharp_fwd_bck = p_sharp_fwd_fwd;

    // Momentum and sharp momentum at forward end of backward subtree
    Eigen::VectorXd p_bck_fwd = this->z_.p;
    Eigen::VectorXd p_sharp_bck_fwd = p_sharp_fwd_fwd;

    // Momentum and sharp momentum at backward end of backward subtree
    Eigen::VectorXd p_bck_bck = this->z_.p;
    Eigen::VectorXd p_sharp_bck_bck = p_sharp_fwd_fwd;

    // Integrated momenta along trajectory
    Eigen::VectorXd rho = this->z_.p.transpose();

    // Log sum of state weights (offset by H0) along trajectory
    double log_sum_weight = 0;  // log(exp(H0 - H0))
    double H0 = this->hamiltonian_.H(this->z_);
    int n_leapfrog = 0;
    double sum_metro_prob = 0;

    // Build a trajectory until the no-u-turn
    // criterion is no longer satisfied
    this->depth_ = 0;
    this->divergent_ = false;

    while (this->depth_ < this->max_depth_) {
      // Build a new subtree in a random direction
      Eigen::VectorXd rho_fwd = Eigen::VectorXd::Zero(rho.size());
      Eigen::VectorXd rho_bck = Eigen::VectorXd::Zero(rho.size());

      bool valid_subtree = false;
      double log_sum_weight_subtree = -std::numeric_limits<double>::infinity();

      if (this->rand_uniform_() > 0.5) {
        // Extend the current trajectory forward
        this->z_.ps_point::operator=(z_fwd);
        rho_bck = rho;
        p_bck_fwd = p_fwd_fwd;
        p_sharp_bck_fwd = p_sharp_fwd_fwd;

        valid_subtree = build_tree(
            this->depth_, z_propose, p_sharp_fwd_bck, p_sharp_fwd_fwd, rho_fwd,
            p_fwd_bck, p_fwd_fwd, H0, 1, n_leapfrog, log_sum_weight_subtree,
            sum_metro_prob, logger);
        z_fwd.ps_point::operator=(this->z_);
      } else {
        // Extend the current trajectory backwards
        this->z_.ps_point::operator=(z_bck);
        rho_fwd = rho;
        p_fwd_bck = p_bck_bck;
        p_sharp_fwd_bck = p_sharp_bck_bck;

        valid_subtree = build_tree(
            this->depth_, z_propose, p_sharp_bck_fwd, p_sharp_bck_bck, rho_bck,
            p_bck_fwd, p_bck_bck, H0, -1, n_leapfrog, log_sum_weight_subtree,
            sum_metro_prob, logger);
        z_bck.ps_point::operator=(this->z_);
      }

      if (!valid_subtree)
        break;

      // Sample from accepted subtree
      ++(this->depth_);

      if (log_sum_weight_subtree > log_sum_weight) {
        z_sample = z_propose;
      } else {
        double accept_prob = std::exp(log_sum_weight_subtree - log_sum_weight);
        if (this->rand_uniform_() < accept_prob)
          z_sample = z_propose;
      }

      log_sum_weight
          = math::log_sum_exp(log_sum_weight, log_sum_weight_subtree);

      // Break when no-u-turn criterion is no longer satisfied
      rho = rho_bck + rho_fwd;

      // Demand satisfaction around merged subtrees
      bool persist_criterion
          = compute_criterion(p_sharp_bck_bck, p_sharp_fwd_fwd, rho);

      // Demand satisfaction between subtrees
      Eigen::VectorXd rho_extended = rho_bck + p_fwd_bck;

      persist_criterion
          &= compute_criterion(p_sharp_bck_bck, p_sharp_fwd_bck, rho_extended);

      rho_extended = rho_fwd + p_bck_fwd;
      persist_criterion
          &= compute_criterion(p_sharp_bck_fwd, p_sharp_fwd_fwd, rho_extended);

      if (!persist_criterion)
        break;
    }

    this->n_leapfrog_ = n_leapfrog;

    // Compute average acceptance probabilty across entire trajectory,
    // even over subtrees that may have been rejected
    double accept_prob = sum_metro_prob / static_cast<double>(n_leapfrog);

    this->z_.ps_point::operator=(z_sample);
    this->energy_ = this->hamiltonian_.H(this->z_);
    return sample(this->z_.q, -this->z_.V, accept_prob);
  }

  void get_sampler_param_names(std::vector<std::string>& names) {
    names.push_back("stepsize__");
    names.push_back("treedepth__");
    names.push_back("n_leapfrog__");
    names.push_back("divergent__");
    names.push_back("energy__");
  }

  void get_sampler_params(std::vector<double>& values) {
    values.push_back(this->epsilon_);
    values.push_back(this->depth_);
    values.push_back(this->n_leapfrog_);
    values.push_back(this->divergent_);
    values.push_back(this->energy_);
  }

  virtual bool compute_criterion(Eigen::VectorXd& p_sharp_minus,
                                 Eigen::VectorXd& p_sharp_plus,
                                 Eigen::VectorXd& rho) {
    return p_sharp_plus.dot(rho) > 0 && p_sharp_minus.dot(rho) > 0;
  }

  /**
   * Recursively build a new subtree to completion or until
   * the subtree becomes invalid.  Returns validity of the
   * resulting subtree.
   *
   * @param depth Depth of the desired subtree
   * @param z_propose State proposed from subtree
   * @param p_sharp_beg Sharp momentum at beginning of new tree
   * @param p_sharp_end Sharp momentum at end of new tree
   * @param rho Summed momentum across trajectory
   * @param p_beg Momentum at beginning of returned tree
   * @param p_end Momentum at end of returned tree
   * @param H0 Hamiltonian of initial state
   * @param sign Direction in time to built subtree
   * @param n_leapfrog Summed number of leapfrog evaluations
   * @param log_sum_weight Log of summed weights across trajectory
   * @param sum_metro_prob Summed Metropolis probabilities across trajectory
   * @param logger Logger for messages
   */
  bool build_tree(int depth, ps_point& z_propose, Eigen::VectorXd& p_sharp_beg,
                  Eigen::VectorXd& p_sharp_end, Eigen::VectorXd& rho,
                  Eigen::VectorXd& p_beg, Eigen::VectorXd& p_end, double H0,
                  double sign, int& n_leapfrog, double& log_sum_weight,
                  double& sum_metro_prob, callbacks::logger& logger) {
    // Base case
    if (depth == 0) {
      this->integrator_.evolve(this->z_, this->hamiltonian_,
                               sign * this->epsilon_, logger);
      ++n_leapfrog;

      double h = this->hamiltonian_.H(this->z_);
      if (std::isnan(h))
        h = std::numeric_limits<double>::infinity();

      if ((h - H0) > this->max_deltaH_)
        this->divergent_ = true;

      log_sum_weight = math::log_sum_exp(log_sum_weight, H0 - h);

      if (H0 - h > 0)
        sum_metro_prob += 1;
      else
        sum_metro_prob += std::exp(H0 - h);

      z_propose = this->z_;

      p_sharp_beg = this->hamiltonian_.dtau_dp(this->z_);
      p_sharp_end = p_sharp_beg;

      rho += this->z_.p;
      p_beg = this->z_.p;
      p_end = p_beg;

      return !this->divergent_;
    }
    // General recursion

    // Build the initial subtree
    double log_sum_weight_init = -std::numeric_limits<double>::infinity();

    // Momentum and sharp momentum at end of the initial subtree
    Eigen::VectorXd p_init_end(this->z_.p.size());
    Eigen::VectorXd p_sharp_init_end(this->z_.p.size());

    Eigen::VectorXd rho_init = Eigen::VectorXd::Zero(rho.size());

    bool valid_init
        = build_tree(depth - 1, z_propose, p_sharp_beg, p_sharp_init_end,
                     rho_init, p_beg, p_init_end, H0, sign, n_leapfrog,
                     log_sum_weight_init, sum_metro_prob, logger);

    if (!valid_init)
      return false;

    // Build the final subtree
    ps_point z_propose_final(this->z_);

    double log_sum_weight_final = -std::numeric_limits<double>::infinity();

    // Momentum and sharp momentum at beginning of the final subtree
    Eigen::VectorXd p_final_beg(this->z_.p.size());
    Eigen::VectorXd p_sharp_final_beg(this->z_.p.size());

    Eigen::VectorXd rho_final = Eigen::VectorXd::Zero(rho.size());

    bool valid_final
        = build_tree(depth - 1, z_propose_final, p_sharp_final_beg, p_sharp_end,
                     rho_final, p_final_beg, p_end, H0, sign, n_leapfrog,
                     log_sum_weight_final, sum_metro_prob, logger);

    if (!valid_final)
      return false;

    // Multinomial sample from right subtree
    double log_sum_weight_subtree
        = math::log_sum_exp(log_sum_weight_init, log_sum_weight_final);
    log_sum_weight = math::log_sum_exp(log_sum_weight, log_sum_weight_subtree);

    if (log_sum_weight_final > log_sum_weight_subtree) {
      z_propose = z_propose_final;
    } else {
      double accept_prob
          = std::exp(log_sum_weight_final - log_sum_weight_subtree);
      if (this->rand_uniform_() < accept_prob)
        z_propose = z_propose_final;
    }

    Eigen::VectorXd rho_subtree = rho_init + rho_final;
    rho += rho_subtree;

    // Demand satisfaction around merged subtrees
    bool persist_criterion
        = compute_criterion(p_sharp_beg, p_sharp_end, rho_subtree);

    // Demand satisfaction between subtrees
    rho_subtree = rho_init + p_final_beg;
    persist_criterion
        &= compute_criterion(p_sharp_beg, p_sharp_final_beg, rho_subtree);

    rho_subtree = rho_final + p_init_end;
    persist_criterion
        &= compute_criterion(p_sharp_init_end, p_sharp_end, rho_subtree);

    return persist_criterion;
  }

  int depth_;
  int max_depth_;
  double max_deltaH_;

  int n_leapfrog_;
  bool divergent_;
  double energy_;
};

}  // namespace mcmc
}  // namespace stan
#endif
