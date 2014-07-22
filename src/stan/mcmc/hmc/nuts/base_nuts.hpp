#ifndef STAN__MCMC__BASE__NUTS__BETA
#define STAN__MCMC__BASE__NUTS__BETA

#include <math.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/functions/min.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    struct nuts_util
    {
      // Constants through each recursion
      double log_u;
      double H0;
      int sign;
      
      // Aggregators through each recursion
      int n_tree;
      double sum_prob;
      bool criterion;

    };
    
    // The No-U-Turn Sampler (NUTS).
    
    template <class M, class P, template<class, class> class H, 
    template<class, class> class I, class BaseRNG>
    class base_nuts: public base_hmc<M, P, H, I, BaseRNG>
    {
      
    public:
      
      base_nuts(M &m, BaseRNG& rng, std::ostream* o, std::ostream* e):
      base_hmc<M, P, H, I, BaseRNG>(m, rng, o, e),
      depth_(0), max_depth_(5), max_delta_(1000), n_leapfrog_(0), n_divergent_(0)
      {};
      
      ~base_nuts() {};
      
      void set_max_depth(const int d) {
        if(d > 0)
          max_depth_ = d;
      }
      
      void set_max_delta(const double d) {
        max_delta_ = d;
      }
      
      int get_max_depth() { return this->max_depth_; }
      double get_max_delta() { return this->max_delta_; }
      
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
        
        int n_cont = init_sample.cont_params().size();
        
        Eigen::VectorXd rho_init = this->z_.p;
        Eigen::VectorXd rho_plus(n_cont); rho_plus.setZero();
        Eigen::VectorXd rho_minus(n_cont); rho_minus.setZero();
        
        util.H0 = this->hamiltonian_.H(this->z_);
        
        // Sample the slice variable
        util.log_u = std::log(this->rand_uniform_());
        
        // Build a balanced binary tree until the NUTS criterion fails
        util.criterion = true;
        int n_valid = 0;
        
        this->depth_ = 0;
        this->n_divergent_ = 0;
        
        util.n_tree = 0;
        util.sum_prob = 0;
        
        while (util.criterion && (this->depth_ <= this->max_depth_) ) {
          
          // Randomly sample a direction in time
          ps_point* z = 0;
          Eigen::VectorXd* rho = 0;
          
          if (this->rand_uniform_() > 0.5) {
            z = &z_plus;
            rho = &rho_plus;
            util.sign = 1;
          }
          else {
            z = &z_minus;
            rho = &rho_minus;
            util.sign = -1;
          }
          
          // And build a new subtree in that direction 
          this->z_.ps_point::operator=(*z);
          
          int n_valid_subtree = build_tree(depth_, *rho, 0, z_propose, util);
          ++(this->depth_);
          
          *z = this->z_;

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
          
          // Check validity of completed tree
          this->z_.ps_point::operator=(z_plus);
          Eigen::VectorXd delta_rho = rho_minus + rho_init + rho_plus;

          util.criterion = compute_criterion(z_minus, this->z_, delta_rho);

        }
        
        this->n_leapfrog_ = util.n_tree;
        
        double accept_prob = util.sum_prob / static_cast<double>(util.n_tree);
        
        this->z_.ps_point::operator=(z_sample);
        return sample(this->z_.q, - this->z_.V, accept_prob);
        
      }
      
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

      virtual bool compute_criterion(ps_point& start, P& finish, Eigen::VectorXd& rho) = 0;
      
      // Returns number of valid points in the completed subtree
      int build_tree(int depth, Eigen::VectorXd& rho, 
                     ps_point* z_init_parent, ps_point& z_propose,
                     nuts_util& util)
      {
        
        // Base case
        if (depth == 0) 
        {
          
          this->integrator_.evolve(this->z_, this->hamiltonian_, 
                                   util.sign * this->epsilon_);
          
          rho += this->z_.p;
          
          if (z_init_parent) *z_init_parent = this->z_;
          z_propose = this->z_;
          
          double h = this->hamiltonian_.H(this->z_); 
          if (boost::math::isnan(h)) h = std::numeric_limits<double>::infinity();
          
          util.criterion = util.log_u + (h - util.H0) < this->max_delta_;
          if (!util.criterion) ++(this->n_divergent_);

          util.sum_prob += stan::math::min(1, std::exp(util.H0 - h));
          util.n_tree += 1;
          
          return (util.log_u + (h - util.H0) < 0);
          
        } 
        // General recursion
        else 
        {
          
          Eigen::VectorXd left_subtree_rho(rho.size()); left_subtree_rho.setZero();
          ps_point z_init(this->z_);
          
          int n1 = build_tree(depth - 1, left_subtree_rho, &z_init, z_propose, util);

          if (z_init_parent) *z_init_parent = z_init;
          
          if (!util.criterion) return 0;
          
          Eigen::VectorXd right_subtree_rho(rho.size()); right_subtree_rho.setZero();
          ps_point z_propose_right(z_init);
          
          int n2 = build_tree(depth - 1, right_subtree_rho, 0, z_propose_right, util);
          
          double accept_prob = static_cast<double>(n2) /
                               static_cast<double>(n1 + n2);
          
          if ( util.criterion && (this->rand_uniform_() < accept_prob) )
            z_propose = z_propose_right;
          
          Eigen::VectorXd& subtree_rho = left_subtree_rho;
          subtree_rho += right_subtree_rho;
          
          rho += subtree_rho;
          
          util.criterion &= compute_criterion(z_init, this->z_, subtree_rho);
          
          return n1 + n2;
          
        }
        
      }

      int depth_;
      int max_depth_;
      double max_delta_;

      int n_leapfrog_;
      int n_divergent_;
      
    };
    
  } // mcmc
  
} // stan


#endif
