#ifndef __STAN__MCMC__BASE__NUTS__BETA__
#define __STAN__MCMC__BASE__NUTS__BETA__

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
      _depth(0), _max_depth(5), _max_delta(1000)
      {};
      
      ~base_nuts() {};
      
      void set_max_depth(const int d) {
        if(d > 0)
          _max_depth = d;
      }
      
      void set_max_delta(const double d) {
        _max_delta = d;
      }
      
      int get_max_depth() { return this->_max_depth; }
      double get_max_delta() { return this->_max_delta; }
      
      sample transition(sample& init_sample)
      {
        
        // Initialize the algorithm
        this->_sample_stepsize();
        
        nuts_util util;
        
        this->seed(init_sample.cont_params(), init_sample.disc_params());
        
        this->_hamiltonian.sample_p(this->_z, this->_rand_int);
        this->_hamiltonian.init(this->_z);

        ps_point z_plus(this->_z);
        ps_point z_minus(z_plus);

        ps_point z_sample(z_plus);
        ps_point z_propose(z_plus);
        
        int n_cont = init_sample.cont_params().size();
        
        Eigen::VectorXd rho_init = this->_z.p;
        Eigen::VectorXd rho_plus(n_cont); rho_plus.setZero();
        Eigen::VectorXd rho_minus(n_cont); rho_minus.setZero();
        
        util.H0 = this->_hamiltonian.H(this->_z);
        
        // Sample the slice variable
        util.log_u = std::log(this->_rand_uniform());
        
        // Build a balanced binary tree until the NUTS criterion fails
        util.criterion = true;
        int n_valid = 0;
        
        this->_depth = 0;
        
        while (util.criterion && (this->_depth <= this->_max_depth) ) {
          
          util.n_tree = 0;
          util.sum_prob = 0;
          
          // Randomly sample a direction in time
          ps_point* z = 0;
          Eigen::VectorXd* rho = 0;
          
          if (this->_rand_uniform() > 0.5)
          {
            z = &z_plus;
            rho = &rho_plus;
            util.sign = 1;
          }
          else
          {
            z = &z_minus;
            rho = &rho_minus;
            util.sign = -1;
          }
          
          // And build a new subtree in that direction 
          this->_z.ps_point::operator=(*z);
          
          int n_valid_subtree = build_tree(_depth, *rho, 0, z_propose, util);
          
          *z = this->_z;

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
          
          if (this->_rand_uniform() < subtree_prob)
            z_sample = z_propose;
          
          n_valid += n_valid_subtree;
          
          // Check validity of completed tree
          this->_z.ps_point::operator=(z_plus);
          Eigen::VectorXd delta_rho = rho_minus + rho_init + rho_plus;
          
          util.criterion = _compute_criterion(z_minus, this->_z, delta_rho);
          
          ++(this->_depth);
          
        }
        
        --(this->_depth); // Correct for increment at end of loop
        
        double accept_prob = util.sum_prob / static_cast<double>(util.n_tree);
        
        this->_z.ps_point::operator=(z_sample);
        return sample(this->_z.q, this->_z.r, - this->_z.V, accept_prob);
        
      }
      
      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,treedepth__,";
      }
      
      void write_sampler_params(std::ostream& o) {
        o << this->_epsilon << "," << this->_depth << ",";
      }
      
      void get_sampler_param_names(std::vector<std::string>& names) {
        names.push_back("stepsize__");
        names.push_back("treedepth__");
      }
      
      void get_sampler_params(std::vector<double>& values) {
        values.push_back(this->_epsilon);
        values.push_back(this->_depth);
      }

      
    protected:
      
      virtual bool _compute_criterion(ps_point& start, P& finish, Eigen::VectorXd& rho) = 0;
      
      // Returns number of valid points in the completed subtree
      int build_tree(int depth, Eigen::VectorXd& rho, 
                     ps_point* z_init, ps_point& z_propose, nuts_util& util)
      {
        
        // Base case
        if (depth == 0) 
        {
          
          this->_integrator.evolve(this->_z, this->_hamiltonian, 
                                   util.sign * this->_epsilon);
          
          rho += this->_z.p;
          
          if (z_init) *z_init = this->_z;
          z_propose = this->_z;
          
          double h = this->_hamiltonian.H(this->_z); 
          if (boost::math::isnan(h)) h = std::numeric_limits<double>::infinity();
          
          util.criterion = util.log_u + (h - util.H0) < this->_max_delta;

          util.sum_prob += stan::math::min(1, std::exp(util.H0 - h));
          util.n_tree += 1;
          
          return (util.log_u + (h - util.H0) < 0);
          
        } 
        // General recursion
        else 
        {
          
          Eigen::VectorXd subtree_rho(rho.size()); subtree_rho.setZero();
          ps_point z_init(this->_z);
          
          int n1 = build_tree(depth - 1, subtree_rho, &z_init, z_propose, util);

          rho += subtree_rho;
          
          if (!util.criterion) return 0;
          
          subtree_rho.setZero();
          ps_point z_propose_right(z_init);
          
          int n2 = build_tree(depth - 1, subtree_rho, 0, z_propose_right, util);
          
          rho += subtree_rho;
          
          double accept_prob = static_cast<double>(n2) /
                               static_cast<double>(n1 + n2);
          
          if ( util.criterion && (this->_rand_uniform() < accept_prob) )
            z_propose = z_propose_right;
          
          util.criterion &= _compute_criterion(z_init, this->_z, rho);
          
          return n1 + n2;
          
        }
        
      }

      int _depth;
      int _max_depth;
      double _max_delta;
      
    };
    
  } // mcmc
  
} // stan


#endif