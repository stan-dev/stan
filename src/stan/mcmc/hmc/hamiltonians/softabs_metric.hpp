#ifndef __STAN__MCMC__SOFTABS__METRIC__BETA__
#define __STAN__MCMC__SOFTABS__METRIC__BETA__

#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

#include <stan/agrad/autodiff.hpp>

#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    template <typename M>
    struct _softabs_fun {
      
      const M& _model;
      std::ostream* _o;
      
      _softabs_fun(M& m, std::ostream* out): _model(m), _o(out) {};
      
      template <typename T>
      T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) const {
        Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,1> > eigen_x(&(x[0]), x.size());
        std::vector<int> dummy_int;
        return _model.log_prob<true,true,T>(eigen_x, dummy_int, _o);
      }
    };
    
    // Riemannian manifold with SoftAbs metric
    template <typename M, typename BaseRNG>
    class softabs_metric: public base_hamiltonian<M, softabs_point, BaseRNG> {
      
    public:
      
      softabs_metric(M& m, std::ostream* e):
        base_hamiltonian<M, softabs_point, BaseRNG>(m, e),
        _alpha(1.0)
      {};
      ~softabs_metric() {};
      
      double T(softabs_point& z) {
        compute_metric(z);
        return this->tau(z) + 0.5 * z.log_det_metric;
      }
      
      // Uses current (possibly stale) metric info in z
      double tau(softabs_point& z) {
        return 0.5 * z.Q_p.dot(z.lambda_Q_p);
      }
      
      // Uses current (possibly stale) metric info in z
      double phi(softabs_point& z) {
        return this->V(z) + 0.5 * z.log_det_metric;
      }
      
      const Eigen::VectorXd dtau_dq(softabs_point& z) {
        z.aux_one.noalias() = z.lambda_Q_p.asDiagonal() * z.eigen_deco.eigenvectors().transpose();
        z.aux_two.noalias() = z.pseudo_j.selfadjointView<Eigen::Lower>() * z.aux_one;
        
        z.cache.setZero();
        z.cache.triangularView<Eigen::Lower>() = z.aux_one.transpose() * z.aux_two;
        
        Eigen::VectorXd aux(z.q.size());
        Eigen::Map<Eigen::VectorXd> eigen_q(&(z.q[0]), z.q.size());
        stan::agrad::grad_tr_mat_times_hessian(_softabs_fun<M>(this->_model, 0), eigen_q, z.cache, aux);
        aux *= -1;
        
        return -0.5 * aux;
      }

      const Eigen::VectorXd dtau_dp(softabs_point& z) {
        return z.eigen_deco.eigenvectors() * z.lambda_Q_p;
      }
      
      const Eigen::VectorXd dphi_dq(softabs_point& z) {
        Eigen::VectorXd aux = z.softabs_lambda_inv.cwiseProduct(z.pseudo_j.diagonal());
        z.aux_two.noalias() = aux.asDiagonal() * z.eigen_deco.eigenvectors().transpose();
        
        z.cache.setZero();
        z.cache.triangularView<Eigen::Lower>() = z.eigen_deco.eigenvectors() * z.aux_two;

        Eigen::Map<Eigen::VectorXd> eigen_q(&(z.q[0]), z.q.size());
        stan::agrad::grad_tr_mat_times_hessian(_softabs_fun<M>(this->_model, 0), eigen_q, z.cache, aux);
        aux *= -1;
        
        return 0.5 * aux + z.g;
        
      }
      
      void sample_p(softabs_point& z, BaseRNG& rng) {
        
        compute_metric(z);
        
        boost::variate_generator<BaseRNG&, boost::normal_distribution<> > 
          _rand_unit_gaus(rng, boost::normal_distribution<>());
        
        Eigen::VectorXd aux(z.p.size());
        
        for (size_t i = 0; i < z.p.size(); ++i)
          aux(i) = sqrt(z.softabs_lambda(i)) * _rand_unit_gaus();

        z.p.noalias() = z.eigen_deco.eigenvectors() * aux;
        
      }
      
      void compute_metric(softabs_point& z)
      {
        
        // Compute the Hessian
        Eigen::Map<Eigen::VectorXd> eigen_q(&(z.q[0]), z.q.size());
        stan::agrad::hessian(_softabs_fun<M>(this->_model, 0), eigen_q, z.V, z.g, z.hessian);
        
        z.V *= -1;
        z.g *= -1;
        
        // Compute the eigen decomposition of the Hessian,
        // then perform the SoftAbs transformation
        z.eigen_deco.compute(z.hessian);
        
        for (size_t i = 0; i < z.q.size(); ++i) {
          
          const double lambda = z.eigen_deco.eigenvalues()(i);
          const double alpha_lambda = _alpha * lambda;
          
          double softabs_lambda = 0;
          
          // Thresholds defined such that the approximation
          // error is on the same order of double precision
          if(std::fabs(alpha_lambda) < 1e-4)
          {
            softabs_lambda = (1.0 + (1.0 / 3.0) * alpha_lambda * alpha_lambda) / _alpha;
          }
          else if(std::fabs(alpha_lambda) > 18)
          {
            softabs_lambda = std::fabs(lambda);
          }
          else
          {
            softabs_lambda = lambda / std::tanh(alpha_lambda);
          }
          
          z.softabs_lambda(i) = softabs_lambda;
          z.softabs_lambda_inv(i) = 1.0 / softabs_lambda;
        }
        
        // Helpful auxiliary calcs
        update_p(z);
        
        // Compute the log determinant of the metric
        z.log_det_metric = 0;
        for (size_t i = 0; i < z.q.size(); ++i)
          z.log_det_metric += std::log(z.softabs_lambda(i));
          
      }
      
      /// Compute intermediate values necessary for the spatial gradients dtau_dq and dphi_dq
      void prepare_spatial_gradients(softabs_point& z)
      {
        
        // Compute the pseudo-Jacobian of the SoftAbs transform
        double delta = 0;
        double lambda = 0;
        double alpha_lambda = 0;
        double sdx = 0;
        
        for (size_t i = 0; i < z.q.size(); ++i) {
          
          for (size_t j = 0; j <= i; ++j) {
            
            delta = z.eigen_deco.eigenvalues()(i) - z.eigen_deco.eigenvalues()(j);
            
            if(std::fabs(delta) < 1e-10)
            {
              
              lambda = z.eigen_deco.eigenvalues()(i);
              alpha_lambda = _alpha * lambda;
              
              // Thresholds defined such that the approximation
              // error is on the same order of double precision
              if(std::fabs(alpha_lambda) < 1e-4)
              {
                z.pseudo_j(i, j) = (2.0 / 3.0) * alpha_lambda
                * (1.0 - (2.0 / 15.0) * alpha_lambda * alpha_lambda);
              }
              else if(std::fabs(alpha_lambda) > 18)
              {
                z.pseudo_j(i, j) = lambda > 0 ? 1 : -1;
              }
              else
              {
                sdx = std::sinh(_alpha * lambda) / lambda;
                z.pseudo_j(i, j) = (z.softabs_lambda(i) - _alpha / (sdx * sdx) ) / lambda;
              }
              
            }
            else
            {
              z.pseudo_j(i, j) = ( z.softabs_lambda(i) - z.softabs_lambda(j) ) / delta;
            }
            
          } // j
          
        } // i
        
      } // prepare_spatial_gradients
      
      void update(softabs_point& z) {
        //base_hamiltonian::update(z);
        compute_metric(z);
        prepare_spatial_gradients(z);
      }
      
      void update_p(softabs_point& z) {
        z.Q_p.noalias() = z.eigen_deco.eigenvectors().transpose() * z.p;
        z.lambda_Q_p.noalias() = z.softabs_lambda_inv.cwiseProduct(z.Q_p);
      }
      
      // Return the product of the current metric inverse with v
      const Eigen::VectorXd metric_inv_dot(softabs_point& z, const Eigen::VectorXd& v) {
        Eigen::VectorXd a = z.eigen_deco.eigenvectors().transpose() * v;
        Eigen::VectorXd b = z.softabs_lambda_inv.cwiseProduct(a);
        return z.eigen_deco.eigenvectors() * b;
      }
      
      double get_alpha() { return this->_alpha; }
      
      virtual void set_alpha(const double a) {
        if(a > 0) this->_alpha = a;
      }
      
    private:
      
      double _alpha;
      
      
    };

    
  } // mcmc
  
} // stan


#endif
