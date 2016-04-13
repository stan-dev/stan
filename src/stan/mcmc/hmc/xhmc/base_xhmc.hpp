#ifndef STAN_MCMC_HMC_NUTS_BASE_XHMC_HPP
#define STAN_MCMC_HMC_NUTS_BASE_XHMC_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
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
     * Exhaustive Hamiltonian Monte Carlo (XHMC) with multinomial sampling.
     * See http://arxiv.org/abs/1601.00225.
     */
    template <class Model, template<class, class> class Hamiltonian,
              template<class> class Integrator, class BaseRNG>
    class base_xhmc : public base_hmc<Model, Hamiltonian, Integrator, BaseRNG> {
    public:
      base_xhmc(const Model& model, BaseRNG& rng)
        : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng),
          depth_(0), max_depth_(5), max_deltaH_(1000), x_delta_(0.1),
          n_leapfrog_(0), divergent_(0), energy_(0) {
      }

      ~base_xhmc() {}

      void set_max_depth(int d) {
        if (d > 0)
          max_depth_ = d;
      }

      void set_max_deltaH(double d) {
        max_deltaH_ = d;
      }

      void set_x_delta(double d) {
      if (d > 0)
        x_delta_ = d;
      }

      int get_max_depth() { return this->max_depth_; }
      double get_max_deltaH() { return this->max_deltaH_; }
      double get_x_delta() { return this->x_delta_; }

      sample transition(sample& init_sample,
                        interface_callbacks::writer::base_writer& writer) {
        // Initialize the algorithm
        this->sample_stepsize();

        this->seed(init_sample.cont_params());

        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_, writer);

        ps_point z_plus(this->z_);
        ps_point z_minus(z_plus);

        ps_point z_sample(z_plus);
        ps_point z_propose(z_plus);

        double sum_numer = this->hamiltonian_.dG_dt(this->z_);
        double sum_weight = 1;

        double H0 = this->hamiltonian_.H(this->z_);
        int n_leapfrog = 0;
        double sum_metro_prob = 1;  // exp(H0 - H0)

        // Build a trajectory until the NUTS criterion is no longer satisfied
        this->depth_ = 0;
        this->divergent_ = 0;

        while (this->depth_ < this->max_depth_) {
          // Build a new subtree in a random direction
          bool valid_subtree = false;
          double sum_numer_subtree = 0;
          double sum_weight_subtree = 0;

          if (this->rand_uniform_() > 0.5) {
            this->z_.ps_point::operator=(z_plus);
            valid_subtree
              = build_tree(this->depth_, z_propose,
                           sum_numer_subtree, sum_weight_subtree,
                           H0, 1, n_leapfrog, sum_metro_prob, writer);
            z_plus.ps_point::operator=(this->z_);
          } else {
            this->z_.ps_point::operator=(z_minus);
            valid_subtree
              = build_tree(this->depth_, z_propose,
                           sum_numer_subtree, sum_weight_subtree,
                           H0, -1, n_leapfrog, sum_metro_prob, writer);
            z_minus.ps_point::operator=(this->z_);
          }

          sum_numer += sum_numer_subtree;
          sum_weight += sum_weight_subtree;
          if (!valid_subtree) break;

          // Sample from an accepted subtree
          ++(this->depth_);

          double accept_prob = sum_weight_subtree / sum_weight;
          if (this->rand_uniform_() < accept_prob)
            z_sample = z_propose;

            // Break if exhaustion criterion is satisfied
            if (std::fabs(sum_numer / sum_weight) < x_delta_)
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

      // Returns number of valid points in the completed subtree
      int build_tree(int depth, ps_point& z_propose,
                     double& sum_numer, double& sum_weight,
                     double H0, double sign, int& n_leapfrog,
                     double& sum_metro_prob,
                     interface_callbacks::writer::base_writer& writer) {
        // Base case
        if (depth == 0) {
            this->integrator_.evolve(this->z_, this->hamiltonian_,
                                     sign * this->epsilon_,
                                     writer);
            ++n_leapfrog;

            double h = this->hamiltonian_.H(this->z_);
            if (boost::math::isnan(h))
              h = std::numeric_limits<double>::infinity();

            if ((h - H0) > this->max_deltaH_) this->divergent_ = true;

            double pi = exp(H0 - h);
            sum_numer += pi * this->hamiltonian_.dG_dt(this->z_);
            sum_weight += pi;
            sum_metro_prob += pi > 1 ? 1 : pi;

            z_propose = this->z_;

            return !this->divergent_;
        }
        // General recursion

        // Build the left subtree
        double sum_numer_left = 0;
        double sum_weight_left = 0;

        bool valid_left
          = build_tree(depth - 1, z_propose,
                       sum_numer_left, sum_weight_left,
                       H0, sign, n_leapfrog, sum_metro_prob, writer);

        sum_numer += sum_numer_left;
        sum_weight += sum_weight_left;
        if (!valid_left) return false;

        // Build the right subtree
        ps_point z_propose_right(this->z_);
        double sum_numer_right = 0;
        double sum_weight_right = 0;

        bool valid_right
          = build_tree(depth - 1, z_propose_right,
                       sum_numer_right, sum_weight_right,
                       H0, sign, n_leapfrog, sum_metro_prob, writer);

        sum_numer += sum_numer_right;
        sum_weight += sum_weight_right;
        if (!valid_right) return false;

        // Multinomial sample from right subtree
        double accept_prob = sum_weight_right / sum_weight;
        if (this->rand_uniform_() < accept_prob)
          z_propose = z_propose_right;

        double sum_numer_subtree = sum_numer_left + sum_numer_right;
        double sum_weight_subtree = sum_weight_left + sum_weight_right;
        return std::fabs(sum_numer_subtree / sum_weight_subtree) >= x_delta_;
      }

      int depth_;
      int max_depth_;
      double max_deltaH_;
      double x_delta_;

      int n_leapfrog_;
      int divergent_;
      double energy_;
    };

  }  // mcmc
}  // stan
#endif
