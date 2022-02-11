#ifndef STAN_MCMC_HMC_NUTS_BASE_XHMC_HPP
#define STAN_MCMC_HMC_NUTS_BASE_XHMC_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <utility>

namespace stan {
namespace mcmc {

/**
 * Exhaustive Hamiltonian Monte Carlo (XHMC) with multinomial sampling.
 * See http://arxiv.org/abs/1601.00225.
 */
template <class Model, template <class, class> class Hamiltonian,
          template <class> class Integrator, class BaseRNG>
class base_xhmc : public base_hmc<Model, Hamiltonian, Integrator, BaseRNG> {
 public:
  base_xhmc(const Model& model, BaseRNG& rng)
      : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng),
        depth_(0),
        max_depth_(5),
        max_deltaH_(1000),
        x_delta_(0.1),
        n_leapfrog_(0),
        divergent_(0),
        energy_(0) {}

  ~base_xhmc() {}

  void set_max_depth(int d) {
    if (d > 0)
      max_depth_ = d;
  }

  void set_max_deltaH(double d) { max_deltaH_ = d; }

  void set_x_delta(double d) {
    if (d > 0)
      x_delta_ = d;
  }

  int get_max_depth() { return this->max_depth_; }
  double get_max_deltaH() { return this->max_deltaH_; }
  double get_x_delta() { return this->x_delta_; }

  sample transition(sample& init_sample, callbacks::logger& logger) {
    // Initialize the algorithm
    this->sample_stepsize();

    this->seed(init_sample.cont_params());

    this->hamiltonian_.sample_p(this->z_, this->rand_int_);
    this->hamiltonian_.init(this->z_, logger);

    ps_point z_plus(this->z_);
    ps_point z_minus(z_plus);

    ps_point z_sample(z_plus);
    ps_point z_propose(z_plus);

    double ave = this->hamiltonian_.dG_dt(this->z_, logger);
    double log_sum_weight = 0;  // log(exp(H0 - H0))

    double H0 = this->hamiltonian_.H(this->z_);
    int n_leapfrog = 0;
    double sum_metro_prob = 1;  // exp(H0 - H0)

    // Build a trajectory until the NUTS criterion is no longer satisfied
    this->depth_ = 0;
    this->divergent_ = 0;

    while (this->depth_ < this->max_depth_) {
      // Build a new subtree in a random direction
      bool valid_subtree = false;
      double ave_subtree = 0;
      double log_sum_weight_subtree = -std::numeric_limits<double>::infinity();

      if (this->rand_uniform_() > 0.5) {
        this->z_.ps_point::operator=(z_plus);
        valid_subtree = build_tree(this->depth_, z_propose, ave_subtree,
                                   log_sum_weight_subtree, H0, 1, n_leapfrog,
                                   sum_metro_prob, logger);
        z_plus.ps_point::operator=(this->z_);
      } else {
        this->z_.ps_point::operator=(z_minus);
        valid_subtree = build_tree(this->depth_, z_propose, ave_subtree,
                                   log_sum_weight_subtree, H0, -1, n_leapfrog,
                                   sum_metro_prob, logger);
        z_minus.ps_point::operator=(this->z_);
      }

      if (!valid_subtree)
        break;
      std::tie(ave, log_sum_weight) = stable_sum(
          ave, log_sum_weight, ave_subtree, log_sum_weight_subtree);

      // Sample from an accepted subtree
      ++(this->depth_);

      double accept_prob = std::exp(log_sum_weight_subtree - log_sum_weight);
      if (this->rand_uniform_() < accept_prob)
        z_sample = z_propose;

      // Break if exhaustion criterion is satisfied
      if (std::fabs(ave) < x_delta_)
        break;
    }

    this->n_leapfrog_ = n_leapfrog;

    // Compute average acceptance probabilty across entire trajectory,
    // even over subtrees that may have been rejected
    double accept_prob = sum_metro_prob / static_cast<double>(n_leapfrog + 1);

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

  /**
   * Recursively build a new subtree to completion or until
   * the subtree becomes invalid.  Returns validity of the
   * resulting subtree.
   *
   * @param depth Depth of the desired subtree
   * @param z_propose State proposed from subtree
   * @param ave Weighted average of dG/dt across trajectory
   * @param log_sum_weight Log of summed weights across trajectory
   * @param H0 Hamiltonian of initial state
   * @param sign Direction in time to built subtree
   * @param n_leapfrog Summed number of leapfrog evaluations
   * @param sum_metro_prob Summed Metropolis probabilities across trajectory
   * @param logger Logger for messages
   * @return whether built tree is valid
   */
  bool build_tree(int depth, ps_point& z_propose, double& ave,
                  double& log_sum_weight, double H0, double sign,
                  int& n_leapfrog, double& sum_metro_prob,
                  callbacks::logger& logger) {
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

      double dG_dt = this->hamiltonian_.dG_dt(this->z_, logger);

      std::tie(ave, log_sum_weight)
          = stable_sum(ave, log_sum_weight, dG_dt, H0 - h);

      if (H0 - h > 0)
        sum_metro_prob += 1;
      else
        sum_metro_prob += std::exp(H0 - h);

      z_propose = this->z_;

      return !this->divergent_;
    }
    // General recursion

    // Build the left subtree
    double ave_left = 0;
    double log_sum_weight_left = -std::numeric_limits<double>::infinity();

    bool valid_left
        = build_tree(depth - 1, z_propose, ave_left, log_sum_weight_left, H0,
                     sign, n_leapfrog, sum_metro_prob, logger);

    if (!valid_left)
      return false;
    std::tie(ave, log_sum_weight)
        = stable_sum(ave, log_sum_weight, ave_left, log_sum_weight_left);

    // Build the right subtree
    ps_point z_propose_right(this->z_);
    double ave_right = 0;
    double log_sum_weight_right = -std::numeric_limits<double>::infinity();

    bool valid_right = build_tree(depth - 1, z_propose_right, ave_right,
                                  log_sum_weight_right, H0, sign, n_leapfrog,
                                  sum_metro_prob, logger);

    if (!valid_right)
      return false;
    std::tie(ave, log_sum_weight)
        = stable_sum(ave, log_sum_weight, ave_right, log_sum_weight_right);

    // Multinomial sample from right subtree
    double ave_subtree;
    double log_sum_weight_subtree;
    std::tie(ave_subtree, log_sum_weight_subtree) = stable_sum(
        ave_left, log_sum_weight_left, ave_right, log_sum_weight_right);

    double accept_prob
        = std::exp(log_sum_weight_right - log_sum_weight_subtree);
    if (this->rand_uniform_() < accept_prob)
      z_propose = z_propose_right;

    return std::abs(ave_subtree) >= x_delta_;
  }

  /**
   * a1 and a2 are running averages of the form
   *   \f$ a1 =   ( \sum_{n \in N1} w_{n} f_{n} )
   *            / ( \sum_{n \in N1}  w_{n} ) \f$
   *   \f$ a2 =   ( \sum_{n \in N2} w_{n} f_{n} )
   *            / ( \sum_{n \in N2}  w_{n} ) \f$
   * and the weights are the respective normalizing constants
   *   \f$ w1 = \sum_{n \in N1} w_{n} \f$
   *   \f$ w2 = \sum_{n \in N2} w_{n}. \f$
   *
   * This function returns the pooled average
   *   \f$ sum_a =   ( \sum_{n \in N1 \cup N2} w_{n} f_{n} )
   *               / ( \sum_{n \in N1 \cup N2}  w_{n} ) \f$
   * and the pooled weights
   *   \f$ log_sum_w = log(w1 + w2). \f$
   *
   * @param[in] a1 First running average, f1 / w1
   * @param[in] log_w1 Log of first summed weight
   * @param[in] a2 Second running average
   * @param[in] log_w2 Log of second summed weight
   * @return Pair of average of input running averages and log of summed input
   * weights
   */
  static std::pair<double, double> stable_sum(double a1, double log_w1,
                                              double a2, double log_w2) {
    if (log_w2 > log_w1) {
      const double e = std::exp(log_w1 - log_w2);
      return std::make_pair((e * a1 + a2) / (1 + e), log_w2 + std::log1p(e));
    } else {
      const double e = std::exp(log_w2 - log_w1);
      return std::make_pair((a1 + e * a2) / (1 + e), log_w1 + std::log1p(e));
    }
  }

  int depth_;
  int max_depth_;
  double max_deltaH_;
  double x_delta_;

  int n_leapfrog_;
  bool divergent_;
  double energy_;
};

}  // namespace mcmc
}  // namespace stan
#endif
