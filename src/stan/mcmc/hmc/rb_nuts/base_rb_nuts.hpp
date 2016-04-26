#ifndef STAN_MCMC_HMC_NUTS_BASE_RB_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_BASE_RB_NUTS_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace stan {
  namespace mcmc {
    /**
     * The Rao-Blackwellized No-U-Turn sampler (NUTS) with multinomial weights
     */
    template <class Model, template<class, class> class Hamiltonian,
              template<class> class Integrator, class BaseRNG>
    class base_rb_nuts : public base_nuts<Model, Hamiltonian, Integrator, BaseRNG> {
    public:
      base_rb_nuts(const Model& model, BaseRNG& rng)
        : base_nuts<Model, Hamiltonian, Integrator, BaseRNG>(model, rng) {}

      ~base_rb_nuts() {}

      sample
      rb_transition(sample& init_sample,
                    std::vector<sample>& rb_samples,
                    interface_callbacks::writer::base_writer& info_writer,
                    interface_callbacks::writer::base_writer& error_writer) {
        // Initialize the algorithm
        this->sample_stepsize();

        this->seed(init_sample.cont_params());

        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_, info_writer, error_writer);

        ps_point z_plus(this->z_);
        ps_point z_minus(z_plus);

        ps_point z_sample(z_plus);
        ps_point z_propose(z_plus);

        Eigen::VectorXd p_sharp_plus = this->hamiltonian_.dtau_dp(this->z_);
        Eigen::VectorXd p_sharp_minus = this->hamiltonian_.dtau_dp(this->z_);
        Eigen::VectorXd rho = this->z_.p;
        double sum_weight = 1;

        double H0 = this->hamiltonian_.H(this->z_);
        int n_leapfrog = 0;
        double sum_metro_prob = 1;  // exp(H0 - H0)

        rb_samples.clear();
        rb_samples.push_back(sample(this->z_.q, -this->z_.V, 1));

        // Build a trajectory until the NUTS criterion is no longer satisfied
        this->depth_ = 0;
        this->divergent_ = 0;

        while (this->depth_ < this->max_depth_) {
          // Build a new subtree in a random direction
          Eigen::VectorXd rho_subtree(rho.size());
          rho_subtree.setZero();

          bool valid_subtree = false;
          double sum_weight_subtree = 0;

          std::vector<sample> subtree_samples;

          if (this->rand_uniform_() > 0.5) {
            this->z_.ps_point::operator=(z_plus);
            valid_subtree
              = build_tree(this->depth_, rho_subtree, z_propose,
                           H0, 1, n_leapfrog,
                           sum_weight_subtree, sum_metro_prob,
                           subtree_samples, info_writer, error_writer);
            z_plus.ps_point::operator=(this->z_);
            p_sharp_plus = this->hamiltonian_.dtau_dp(this->z_);
          } else {
            this->z_.ps_point::operator=(z_minus);
            valid_subtree
              = build_tree(this->depth_, rho_subtree, z_propose,
                           H0, -1, n_leapfrog,
                           sum_weight_subtree, sum_metro_prob,
                           subtree_samples, info_writer, error_writer);
            z_minus.ps_point::operator=(this->z_);
            p_sharp_minus = this->hamiltonian_.dtau_dp(this->z_);
          }

          sum_weight += sum_weight_subtree;
          if (!valid_subtree) break;

          for (size_t n = 0; n < subtree_samples.size(); ++n)
            rb_samples.push_back(subtree_samples.at(n));

          // Sample from an accepted subtree
          ++(this->depth_);

          double accept_prob = sum_weight_subtree / sum_weight;
          if (this->rand_uniform_() < accept_prob)
            z_sample = z_propose;

          // Break when NUTS criterion is not longer satisfied
          rho += rho_subtree;
          if (!this->compute_criterion(p_sharp_minus, p_sharp_plus, rho))
            break;
        }

        this->n_leapfrog_ = n_leapfrog;

        // Compute average acceptance probabilty across entire trajectory,
        // even over subtrees that may have been rejected
        double accept_prob
          = sum_metro_prob / static_cast<double>(n_leapfrog + 1);

        this->z_.ps_point::operator=(z_sample);
        this->energy_ = this->hamiltonian_.H(this->z_);
        return sample(this->z_.q, -this->z_.V, accept_prob);
      }

      // Returns number of valid points in the completed subtree
      int build_tree(int depth, Eigen::VectorXd& rho, ps_point& z_propose,
                     double H0, double sign, int& n_leapfrog,
                     double& sum_weight, double& sum_metro_prob,
                     std::vector<sample>& subtree_samples,
                     interface_callbacks::writer::base_writer& info_writer,
                     interface_callbacks::writer::base_writer& error_writer) {
        // Base case
        if (depth == 0) {
            this->integrator_.evolve(this->z_, this->hamiltonian_,
                                     sign * this->epsilon_,
                                     info_writer, error_writer);
            ++n_leapfrog;

            double h = this->hamiltonian_.H(this->z_);
            if (boost::math::isnan(h))
              h = std::numeric_limits<double>::infinity();

            if ((h - H0) > this->max_deltaH_) this->divergent_ = true;

            double pi = exp(H0 - h);
            sum_weight += pi;
            sum_metro_prob += pi > 1 ? 1 : pi;

            subtree_samples.push_back(sample(this->z_.q, -this->z_.V, pi));

            z_propose = this->z_;
            rho += this->z_.p;

            return !this->divergent_;
        }
        // General recursion
        Eigen::VectorXd p_sharp_left = this->hamiltonian_.dtau_dp(this->z_);

        Eigen::VectorXd rho_subtree(rho.size());
        rho_subtree.setZero();

        // Build the left subtree
        double sum_weight_left = 0;

        bool valid_left
          = build_tree(depth - 1, rho_subtree, z_propose,
                       H0, sign, n_leapfrog,
                       sum_weight_left, sum_metro_prob,
                       subtree_samples, info_writer, error_writer);

        sum_weight += sum_weight_left;
        if (!valid_left) return false;

        // Build the right subtree
        ps_point z_propose_right(this->z_);
        double sum_weight_right = 0;

        bool valid_right
          = build_tree(depth - 1, rho_subtree, z_propose_right,
                       H0, sign, n_leapfrog,
                       sum_weight_right, sum_metro_prob,
                       subtree_samples, info_writer, error_writer);

        sum_weight += sum_weight_right;
        if (!valid_right) return false;

        // Multinomial sample from right subtree
        double accept_prob = sum_weight_right / sum_weight;
        if (this->rand_uniform_() < accept_prob)
          z_propose = z_propose_right;

        rho += rho_subtree;
        Eigen::VectorXd p_sharp_right = this->hamiltonian_.dtau_dp(this->z_);
        return this->compute_criterion(p_sharp_left, p_sharp_right, rho_subtree);
      }
    };

  }  // mcmc
}  // stan
#endif
