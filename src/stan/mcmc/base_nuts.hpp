#ifndef __STAN__MCMC__BASE__NUTS__BETA__
#define __STAN__MCMC__BASE__NUTS__BETA__

#include <stan/math/util.hpp>
#include <stan/mcmc/base_hmc.hpp>

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
      int n_valid;
      int n_valid_subtree;
      double sum_prob;
      bool criterion;
      
      // Scratch
      std::vector<ps_point> z_init;
    };
    
    
    // The No-U-Turn Sampler (NUTS).
    
    template <class M, class P, template<class, class> class H, 
    template<class, class> class I, class BaseRNG>
    class base_nuts: public base_hmc<M, P, H, I, BaseRNG>
    {
      
    public:
      
      base_nuts(M &m, BaseRNG& rng): base_hmc<M, P, H, I, BaseRNG>(m, rng),
                                     _epsilon(0.1), _depth(0), 
                                     _max_depth(5), _max_delta(1000)
      {};
      
      ~base_nuts() {};
      
      void set_stepsize(const double e) {
        if(e > 0)
          _epsilon = e;
      }
      
      void set_max_depth(const int d) {
        if(d > 0)
          _max_depth = d;
      }
      
      void set_max_delta(const double d) {
        _max_delta = d;
      }
      
      sample transition(sample& init_sample)
      {
        
        // Initialize the algorithm
        nuts_util util;
        
        this->seed(init_sample.cont_params(), init_sample.disc_params());
        
        this->_hamiltonian.sample_p(this->_z, this->_rand_int);
        this->_hamiltonian.init(this->_z);

        ps_point z_plus(dynamic_cast<ps_point>(this->_z));
        ps_point z_minus(dynamic_cast<ps_point>(this->_z));

        ps_point z_propose(dynamic_cast<ps_point>(this->_z));
        
        int n_cont = init_sample.cont_params().size();
        int n_disc = init_sample.disc_params().size();
        
        Eigen::VectorXd rho_plus = Eigen::VectorXd::Zero(n_cont);
        Eigen::VectorXd rho_minus = Eigen::VectorXd::Zero(n_cont);
        
        util.H0 = this->_hamiltonian.H(this->_z);
        
        // Sample the slice variable
        util.log_u = log(this->_rand_uniform()) - util.H0;
        
        // Build a balanced binary tree until the NUTS criterion fails
        util.n_tree = 0;
        util.n_valid = 0;
        util.n_valid_subtree = 0;
        util.sum_prob = 0;
        util.criterion = true;
        
        this->_depth = 0;
        
        while (util.criterion && (this->_depth <= this->_max_depth) ) {
          
          ps_point z_init(n_cont, n_disc);
          util.z_init.push_back(z_init);
          
          // Randomly sample a direction in time
          ps_point& z = z_plus;
          Eigen::VectorXd& rho = rho_plus;
          
          if (this->_rand_uniform() > 0.5)
          {
            z = z_plus;
            rho = rho_plus;
            util.sign = 1;
          }
          else
          {
            z = z_minus;
            rho = rho_minus;
            util.sign = -1;
          }
          
          // And build a new subtree in that direction 
          this->_z.copy_base(z);
          
          util.n_valid_subtree = 0;
          build_tree(_depth, rho, z_propose, 0, util);
          
          z = static_cast<ps_point>(this->_z);
          
          if (!util.criterion) break;
          
          this->_z.copy_base(z_plus);
          
          util.criterion = _compute_criterion(z_minus, this->_z, rho_plus - rho_minus);
          
          // Metropolis-Hastings sample the fresh subtree
          if(util.n_valid)
          {
            if (this->_rand_uniform() < float(util.n_valid_subtree) / float(util.n_valid)) 
              this->_z.copy_base(z_propose);
          }
          else this->_z.copy_base(z_propose);
          
          util.n_valid += util.n_valid_subtree;
          
          ++this->_depth;
          
        }
                                
        double acceptProb = util.sum_prob / static_cast<double>(util.n_tree);
        
        return sample(this->_z.q, this->_z.r, - this->_hamiltonian.V(this->_z), acceptProb);
                                
      }
      
      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,depth__,";
      }
      
      void write_sampler_params(std::ostream& o) {
        o << this->_epsilon << "," << this->_depth << ",";
      }
      
      void get_sampler_param_names(std::vector<std::string>& names) {
        names.clear();
        names.push_back("stepsize__");
        names.push_back("depth__");
      }
      
      void get_sampler_params(std::vector<double>& values) {
        values.clear();
        values.push_back(this->_epsilon);
        values.push_back(this->_depth);
      }

      
    private:
      
      virtual bool _compute_criterion(ps_point& start, P& finish, Eigen::VectorXd& rho) = 0;
      
      void build_tree(int depth, Eigen::VectorXd& rho, 
                      ps_point& z_propose, int n_valid_step, nuts_util& util)
      {
        
        // Base case
        if (depth == 0) 
        {
          
          this->_integrator.evolve(this->_z, this->_hamiltonian, 
                                   util.sign * this->_epsilon);
          
          rho += this->_z.p;
          z_propose = static_cast<ps_point>(this->_z);
          
          double h = this->_hamiltonian.H(this->_z); 
          if(h != h) h = std::numeric_limits<double>::infinity();
          
          util.criterion = util.log_u + h > this->_delta_max;
          
          util.sum_prob += stan::math::min(1, exp(util.H0 - h));
          util.n_tree += 1;
          
          n_valid_step = (h < - util.log_u);
          util.n_valid_subtree += n_valid_step;
          
        } 
        // General recursion
        else 
        {
          
          // Note that the output value of n_valid_step_local
          // isn't used in the first tree expansion
          int n_valid_step_local;
          
          build_tree(depth - 1, rho, z_propose, n_valid_step_local, util);
          
          if(depth == 1) util.z_init.at(depth) = dynamic_cast<ps_point>(this->_z);
          else           util.z_init.at(depth) = util.z_init.at(depth - 1);
          
          if (util.criterion) 
          {
            
            util.n_valid_subtree = 0;
            
            build_tree(depth - 1, rho, z_propose, n_valid_step_local, util);
            
            if (util.criterion && 
                (this->_rand_uniform() < static_cast<double>(n_valid_step_local) / 
                                         static_cast<double>(util.n_valid_subtree))
                ) 
              z_propose = static_cast<ps_point>(this->_z);
            
          }
          
          util.criterion &= _compute_criterion(util.z_init.at(depth), this->_z, rho);
          
        }
        
      }
      
      double _epsilon;
      int _depth;
      int _max_depth;
      double _max_delta;
      
    };
    
  } // mcmc
  
} // stan


#endif