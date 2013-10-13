#ifndef __STAN__MCMC__SOFTABS__POINT__BETA__
#define __STAN__MCMC__SOFTABS__POINT__BETA__

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Point in a phase space where the base manifold
    // is endowed wtih a SoftAbs Riemannian metric
    class softabs_point: public ps_point {
      
    public:
      
      softabs_point(int n, int m):
        ps_point(n, m),
        log_det_metric(0),
        hessian(Eigen::MatrixXd::Identity(n, n)),
        eigen_deco(n),
        softabs_lambda(Eigen::VectorXd::Zero(n)),
        softabs_lambda_inv(Eigen::VectorXd::Zero(n)),
        pseudo_j(Eigen::MatrixXd::Identity(n, n)),
        Q_p(Eigen::VectorXd::Zero(n)),
        lambda_Q_p(Eigen::VectorXd::Zero(n)),
        aux_one(Eigen::MatrixXd::Identity(n, n)),
        aux_two(Eigen::MatrixXd::Identity(n, n)),
        cache(Eigen::MatrixXd::Identity(n, n)),
        fp_init(Eigen::VectorXd::Zero(n)),
        fp_delta(Eigen::VectorXd::Zero(n))
      {};
      
      // All metric-related quantities computed dynamically
      // so no fast copy constructor is required
      
      double log_det_metric;
      
      Eigen::MatrixXd hessian;
      
      // Eigendecomposition of the Hessian
      SelfAdjointEigenSolver<MatrixXd> eigen_deco;
      
      // SoftAbs transformed eigenvalues of Hessian
      Eigen::VectorXd softabs_lambda;
      Eigen::VectorXd softabs_lambda_inv;
      
      // Psuedo-Jacobian of the eigenvalues
      Eigen::MatrixXd psuedo_j;
      
      // Auxilliary members for efficient matrix calculations
      Eigen::VectorXd Q_p;
      Eigen::VectorXd lambda_Q_p;
      
      Eigen::MatrixXd aux_one;
      Eigen::MatrixXd aux_two;
      Eigen::matrixXd cache;

      Eigen::VectorXd fp_init;  // initial point in fixed-point iteration
      Eigen::VectorXd fp_delta; // delta in fixed-point iteration
      
    };
    
  } // mcmc
  
} // stan


#endif
