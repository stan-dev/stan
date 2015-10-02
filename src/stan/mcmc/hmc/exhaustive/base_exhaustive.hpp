#ifndef STAN_MCMC_HMC_NUTS_BASE_EXHAUSTIVE_HPP
#define STAN_MCMC_HMC_NUTS_BASE_EXHAUSTIVE_HPP

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

    template <class M, class P, template<class, class> class H,
              template<class, class> class I, class BaseRNG>
    class base_exhaustive: public base_hmc<M, P, H, I, BaseRNG> {
    private:
      int depth_;
      int max_depth_;
      double max_Delta_;
      double exhaustion_delta_;
      int n_leapfrog_;
      bool divergent_;

    public:
      base_exhaustive(M &m, BaseRNG& rng, std::ostream* o, std::ostream* e)
        : base_hmc<M, P, H, I, BaseRNG>(m, rng, o, e),
        depth_(0), max_depth_(10), max_Delta_(1000),
        exhaustion_delta_(0.1), n_leapfrog_(0), divergent_(false) {
      }

      ~base_exhaustive() {}

      void set_max_depth(int d) {
        if (d > 0)
          max_depth_ = d;
      }

      void set_max_Delta(double d) {
        max_Delta_ = d;
      }

      int get_max_depth() { return this->max_depth_; }
      double get_max_Delta() { return this->max_Delta_; }

      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,treedepth__,n_leapfrog__,divergent__,";
      }

      void write_sampler_params(std::ostream& o) {
        o << this->epsilon_    << "," << this->depth_ << ","
        << this->n_leapfrog_ << "," << this->divergent_ << ",";
      }

      void get_sampler_param_names(std::vector<std::string>& names) {
        names.push_back("stepsize__");
        names.push_back("treedepth__");
        names.push_back("n_leapfrog__");
        names.push_back("divergent__");
      }

      void get_sampler_params(std::vector<double>& values) {
        values.push_back(this->epsilon_);
        values.push_back(this->depth_);
        values.push_back(this->n_leapfrog_);
        values.push_back(this->divergent_);
      }

      // Returns validity of completed subtree
      bool build_tree(int depth,
                      double& ex_numer,
                      double& ex_denom,
                      double H0,
                      double sign,
                      ps_point& z_propose) {
        // Base case
        if (depth == 0) {
          this->integrator_.evolve(this->z_, this->hamiltonian_,
                                   sign * this->epsilon_);

          double h = this->hamiltonian_.H(this->z_);
          if (boost::math::isnan(h))
            h = std::numeric_limits<double>::infinity();

          if ((h - H0) > this->max_Delta_) this->divergent_ = true;

          double pi = exp(H0 - h);
          ex_numer += pi * this->hamiltonian_.dG_dt(this->z_);
          ex_denom += pi;

          z_propose = this->z_;

          return !this->divergent_;

        } else {
          // General recursion
          double ex_numer_left = 0;
          double ex_denom_left = 0;

          // Build the left subtree
          bool valid_left = build_tree(depth - 1,
                                       ex_numer_left, ex_denom_left, H0,
                                       sign, z_propose);

          if (!valid_left) return false;

          // Build the right subtree
          ps_point z_propose_right(this->z_);
          double ex_numer_right = 0;
          double ex_denom_right = 0;

          bool valid_right = build_tree(depth - 1,
                                        ex_numer_right, ex_denom_right, H0,
                                        sign, z_propose_right);

          if (!valid_right) return false;

          ex_numer = ex_numer_left + ex_numer_right;
          ex_denom = ex_denom_left + ex_denom_right;

          double accept_prob = ex_denom_right / ex_denom;
          if (this->rand_uniform_() < accept_prob)
            z_propose = z_propose_right;

          return std::fabs(ex_numer / ex_denom) >= exhaustion_delta_;
        }
      }

      sample transition(sample& init_sample) {
        // Initialize the algorithm
        this->sample_stepsize();

        this->seed(init_sample.cont_params());

        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_);

        ps_point z_plus(this->z_);
        ps_point z_minus(z_plus);

        ps_point z_sample(z_plus);
        ps_point z_propose(z_plus);

        double ex_numer = this->hamiltonian_.dG_dt(this->z_);
        double ex_denom = 1;
        double H0 = this->hamiltonian_.H(this->z_);

        // Build a balanced binary tree until the
        // exhaustion criterion is satsified
        this->depth_ = 0;
        this->divergent_ = false;

        while (this->depth_ < this->max_depth_) {
          // Build a new subtree in a random direction
          bool valid_subtree = false;
          double ex_numer_subtree = 0;
          double ex_denom_subtree = 0;

          if (this->rand_uniform_() > 0.5) {
            this->z_.ps_point::operator=(z_plus);
            valid_subtree = build_tree(this->depth_,
                                       ex_numer_subtree, ex_denom_subtree,
                                       H0, 1, z_propose);
          } else {
            this->z_.ps_point::operator=(z_minus);
            valid_subtree = build_tree(this->depth_,
                                       ex_numer_subtree, ex_denom_subtree,
                                       H0, -1, z_propose);
          }

          if (!valid_subtree) break;

          ++(this->depth_);

          // Sample from an accepted subtree
          ex_numer += ex_numer_subtree;
          ex_denom += ex_denom_subtree;

          double accept_prob = ex_denom_subtree / ex_denom;

          if (this->rand_uniform_() < accept_prob)
            z_sample = z_propose;

          // Break if exhaustion criterion is satisfied
          if (std::fabs(ex_numer / ex_denom) < exhaustion_delta_)
            break;
        }

        this->n_leapfrog_ = (1 << this->depth_);
        double accept_prob = ex_denom / this->n_leapfrog_;

        this->z_.ps_point::operator=(z_sample);
        return sample(this->z_.q, -this->z_.V, accept_prob);
      }
    };
  }  // mcmc
}  // stan
#endif
