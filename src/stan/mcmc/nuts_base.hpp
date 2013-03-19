#ifndef __STAN__MCMC__NUTS_BASE__BETA__
#define __STAN__MCMC__NUTS_BASE__BETA__

#include <ctime>
#include <iostream>

#include <Eigen/Dense>

#include <stan/mcmc/hamiltonian.hpp>
#include <stan/mcmc/util.hpp>

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
    }

    
    // The No-U-Turn Sampler (NUTS).
  
    template <typename M, typename H, typename I, 
    class BaseRNG = boost::mt19937>
    class nuts_base: public base_hmc<M, H, I, BaseRNG>
    {
        
    public:
            
      nuts_base(double epsilon, M &m);
            
      void sample(Eigen::VectorXd& q);
            
    private:
            
      virtual bool _compute_criterion(psPoint& minus, psPoint& plus, Eigen::VectorXd& rho) = 0;
            
      void build_tree(int depth, psPoint& z, Eigen::VectorXd& rho, Eigen::VectorXd& qPropose,
                      int n_valid_step, nuts_util& util);
     
      double _epsilon;
        
    }

    nuts_base::nuts_base(double epsilon, M& m):
    hmc_base(m),
    _epsilon(epsilon)
    {}

    void nuts_base::sample(Eigen::VectorXd& q)
    {
        
      // Initialize the algorithm
      nuts_util util;
      
      psPoint z(model.dim());
      z.q = q;
      _hamiltonian.sampleP(z.p, RNG);
      
      psPoint zPlus(z);
      psPoint zMinus(z);
      
      Eigen::VectorXd rhoPlus = Eigen::VectorXd::Zero(model.dim());
      Eigen::VectorXd rhoMinus = Eigen::VectorXd::Zero(model.dim());
      
      Eigen::VectorXd qPropose(model.dim());
      
      util.H0 = _hamiltonian.H(z);
      
      // Sample the slice variable
      util.log_u = log(RNG.U()) - util.H0;
      
      // Build a balanced binary tree until the NUTS criterion fails
      util.n_tree = 0;
      util.n_valid = 0;
      util.n_valid_subtree = 0;
      util.sum_prop = 0;
      util.criterion = true;
      
      int depth = 0;
        
      while (util.criterion && depth <= _maxdepth)) 
      {
          
        util.z_init.push_back(psPoint(model.dim()));
        
        // Randomly sample a direction in time
        psPoint& z;
        Eigen::VectorXd& rho;
        
        if (RNG.U() > 0.5)
        {
            z = zPlus;
            rho = rhoPlus;
            util.sign = 1;
        }
        else
        {
            z = zMinus;
            rho = rhoMinus;
            util.sign = -1;
        }
        
        // And build a new subtree in that direction 
        util.n_valid_subtree = 0;
        build_tree(depth, z, rho, qPropose, depth, util); // Second call to depth here is a dummy reference
        
        if (!criterion) break;
        
        util.criterion = _compute_criterion(zMinus, zPlus, rhoPlus - rhoMinus);
        
        // Metropolis-Hastings sample the fresh subtree
        if(util.n_valid)
        {
            if (RNG.U() < float(util.n_valid_subtree) / float(util.n_valid)) 
                q = qPropose;
        }
        else q = qPropose;
        
        util.n_valid += util.n_valid_subtree;
        
        ++depth;
          
      }
      
      return;
        
    }

    void nuts_base::build_tree(int depth, psPoint& z, Eigen::VectorXd& rho, 
                              Eigen::VectorXd& qPropose, int n_valid_step, nuts_util& util)
    {
        
      // Base case
      if (depth == 0) 
      {
          
        _evolve.evolve(z, _Hamiltonian, util.sign * _epsilon);
        
        rho += z.p;
        qPropose = z.q;
        
        double H = point.H();
        if(H != H) H = std::numeric_limits<double>::infinity();
        
        criterion = util.log_u + H > _delta_max;
        
        util.sum_prob += stan::math::min(1, exp(H0 - H));
        util.n_tree += 1;
        
        n_valid_step = (H < - log_u)
        util.n_valid_subtree += n_valid_step;
          
      } 
      // General recursion
      else 
      {
          
        // Note that the output value of n_valid_step_local
        // isn't used in the first tree expansion
        int n_valid_step_local;
        
        build_tree(depth - 1, z, rho, sign, qPropose, n_valid_step_local, util)
        
        if(depth == 1) util.z_init(depth) = z.p();
        else           util.z_init(depth) = util.z_init(depth - 1);
        
        if (criterion) 
        {
            
            n_valid_subtree = 0;
            
            build_tree(depth - 1, z, rho, sign, qPropose, n_valid_step_local, util) 
            
            if (criterion && 
                (RNG.U() < float(n_valid_step_local) / float(util.n_valid_subtree))
                ) 
                qPropose = z.q;
            
        }
        
        util.criterion &= _compute_criterion(z_init(depth), z, rho);
          
      }
        
    }
    
  } // mcmc
  
} // stan


#endif

