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

    struct exhaustive_util {
      // Constants through each recursion
      double log_u;
      double H0;
      int sign;

      // Aggregators through each recursion
      int n_tree;
      double sum_prob;
      bool criterion;
    };


    template <class M, class P, template<class, class> class H,
              template<class, class> class I, class BaseRNG>
    class base_exhaustive: public base_hmc<M, P, H, I, BaseRNG> {
    public:
      base_exhaustive(M &m, BaseRNG& rng, std::ostream* o, std::ostream* e)
        : base_hmc<M, P, H, I, BaseRNG>(m, rng, o, e),
        depth_(0), max_depth_(5), max_delta_(1000),
        exhaustion_delta_(0.1), n_leapfrog_(0), n_divergent_(0) {
      }

      ~base_exhaustive() {}

      void set_max_depth(int d) {
        if (d > 0)
          max_depth_ = d;
      }

      void set_max_delta(double d) {
        max_delta_ = d;
      }

      int get_max_depth() { return this->max_depth_; }
      double get_max_delta() { return this->max_delta_; }

      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,treedepth__,n_leapfrog__,n_divergent__,";
      }

      void write_sampler_params(std::ostream& o) {
        o << this->epsilon_    << "," << this->depth_ << ","
        << this->n_leapfrog_ << "," << this->n_divergent_ << ",";
      }

      void get_sampler_param_names(std::vector<std::string>& names) {
        names.push_back("stepsize__");
        names.push_back("treedepth__");
        names.push_back("n_leapfrog__");
        names.push_back("n_divergent__");
      }

      void get_sampler_params(std::vector<double>& values) {
        values.push_back(this->epsilon_);
        values.push_back(this->depth_);
        values.push_back(this->n_leapfrog_);
        values.push_back(this->n_divergent_);
      }

      bool check_termination(double G_init, double G_final, double delta_t) {
        return std::fabs( (G_final - G_init) / delta_t ) < exhaustion_delta_;
      }

      // Returns number of valid points in the completed subtree
      int build_tree(int depth,
                     double& G_init,
                     double& G_final,
                     ps_point& z_propose,
                     nuts_util& util) {
        // Base case
        if (depth == 0) {
          this->integrator_.evolve(this->z_, this->hamiltonian_,
                                   util.sign * this->epsilon_);

          G_init  = this->z_.p.dot(this->z_.q);
          G_final = G_init;

          z_propose = this->z_;

          double h = this->hamiltonian_.H(this->z_);
          if (boost::math::isnan(h))
            h = std::numeric_limits<double>::infinity();

          util.criterion = util.log_u + (h - util.H0) < this->max_delta_;
          if (!util.criterion) ++(this->n_divergent_);

          util.sum_prob += std::min(1.0, std::exp(util.H0 - h));
          util.n_tree += 1;

          return (util.log_u + (h - util.H0) < 0);

        } else {
          // General recursion
          double G_left = 0;
          double G_right = 0;

          int n1 = build_tree(depth - 1, G_left, G_right, z_propose, util);

          G_init = G_left;
          G_final = G_right;

          if (!util.criterion) return 0;

          ps_point z_propose_right(this->z_);

          int n2 = build_tree(depth - 1, G_left, G_right, z_propose_right, util);

          G_final = G_right;

          double accept_prob =   static_cast<double>(n2)
                               / static_cast<double>(n1 + n2);

          if ( util.criterion && (this->rand_uniform_() < accept_prob) )
            z_propose = z_propose_right;

          util.criterion &= !check_termination(G_init, G_final,
                                               this->epsilon_ * (1 << depth));

          return n1 + n2;
        }
      }

      sample transition(sample& init_sample) {
        // Initialize the algorithm
        this->sample_stepsize();

        nuts_util util;

        this->seed(init_sample.cont_params());

        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_);

        ps_point z_plus(this->z_);
        ps_point z_minus(z_plus);

        ps_point z_sample(z_plus);
        ps_point z_propose(z_plus);

        double G_plus = this->z_.p.dot(this->z_.q);
        double G_minus = G_plus;
        double G_dump;

        util.H0 = this->hamiltonian_.H(this->z_);

        // Sample the slice variable
        util.log_u = std::log(this->rand_uniform_());

        // Build a balanced binary tree until the exhaustion criterion is satsified
        util.criterion = true;
        int n_valid = 0;

        this->depth_ = 0;
        this->n_divergent_ = 0;

        util.n_tree = 0;
        util.sum_prob = 0;

        while (util.criterion && (this->depth_ <= this->max_depth_)) {
          // Build a new subtree in a random direction
          int n_valid_subtree = 0;

          if (this->rand_uniform_() > 0.5) {
            util.sign = 1;
            this->z_.ps_point::operator=(z_plus);
            n_valid_subtree = build_tree(depth_, G_dump, G_plus, z_propose, util);
          } else {
            util.sign = -1;
            this->z_.ps_point::operator=(z_minus);
            n_valid_subtree = build_tree(depth_, G_dump, G_minus, z_propose, util);
          }

          ++(this->depth_);

          // Metropolis-Hastings sample the fresh subtree
          if (!util.criterion)
            break;

          double subtree_prob = 0;

          if (n_valid) {
            subtree_prob = static_cast<double>(n_valid_subtree) /
              static_cast<double>(n_valid);
          } else {
            subtree_prob = n_valid_subtree ? 1 : 0;
          }

          if (this->rand_uniform_() < subtree_prob)
            z_sample = z_propose;

          n_valid += n_valid_subtree;

          util.criterion
            = !check_termination(G_plus, G_minus, this->epsilon_ * util.n_tree);

        }

        this->n_leapfrog_ = util.n_tree;

        double accept_prob = util.sum_prob / static_cast<double>(util.n_tree);

        this->z_.ps_point::operator=(z_sample);
        return sample(this->z_.q, - this->z_.V, accept_prob);
      }

    private:
      int depth_;
      int max_depth_;
      double max_delta_;

      double exhaustion_delta_;

      int n_leapfrog_;
      int n_divergent_;
    };

  }  // mcmc
}  // stan
#endif
