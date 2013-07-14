#ifndef __STAN__MODEL__UTIL_HPP__
#define __STAN__MODEL__UTIL_HPP__

#include <vector>
#include <iostream>

namespace stan {

  namespace model {

    template <class M, bool propto, bool jacobian_adjust_transform>
    void finite_diff_grad(M& model,
                          std::vector<double>& params_r,
                          std::vector<int>& params_i,
                          std::vector<double>& grad,
                          double epsilon = 1e-6,
                          std::ostream* msgs = 0) {
      std::vector<double> perturbed(params_r);
      model
        .template grad_log_prob<propto,
                                jacobian_adjust_transform>(params_r, params_i,
                                                           grad, msgs);
      grad.resize(params_r.size());
      for (size_t k = 0; k < params_r.size(); k++) {
        perturbed[k] += epsilon;
        double logp_plus = model.template log_prob_poly<propto,jacobian_adjust_transform>(perturbed,params_i,msgs);
        perturbed[k] = params_r[k] - epsilon;
        double logp_minus = model.template log_prob_poly<propto,jacobian_adjust_transform>(perturbed,params_i,msgs);
        double gradest = (logp_plus - logp_minus) / (2*epsilon);
        grad[k] = gradest;
        perturbed[k] = params_r[k]; 
      }
    }


    template <class M, bool propto, bool jacobian_adjust_transform>
    int test_gradients(M& model,
                       std::vector<double>& params_r,
                       std::vector<int>& params_i,
                       double epsilon = 1e-6,
                       double error = 1e-6,
                       std::ostream& o = std::cout,
                       std::ostream* msgs = 0) {
      std::vector<double> grad;
      double lp 
        = model.template grad_log_prob<propto,
                                       jacobian_adjust_transform>(params_r,
                                                                  params_i,
                                                                  grad,msgs);
        
      std::vector<double> grad_fd;
      finite_diff_grad<M,propto,
                       jacobian_adjust_transform>(model,
                                                  params_r, params_i,
                                                  grad_fd, epsilon,
                                                  msgs);

      int num_failed = 0;
        
      o << std::endl
        << " Log probability=" << lp
        << std::endl;

      o << std::endl
        << std::setw(10) << "param idx"
        << std::setw(16) << "value"
        << std::setw(16) << "model"
        << std::setw(16) << "finite diff"
        << std::setw(16) << "error" 
        << std::endl;
      for (size_t k = 0; k < params_r.size(); k++) {
        o << std::setw(10) << k
          << std::setw(16) << params_r[k]
          << std::setw(16) << grad[k]
          << std::setw(16) << grad_fd[k]
          << std::setw(16) << (grad[k] - grad_fd[k])
          << std::endl;
        if (std::fabs(grad[k] - grad_fd[k]) > error)
          num_failed++;
      }
      return num_failed;
    }



    template <class M, bool propto, bool jacobian_adjust_transform>
    double grad_hess_log_prob(M& model,
                              std::vector<double>& params_r, 
                              std::vector<int>& params_i,
                              std::vector<double>& gradient,
                              std::vector<double>& hessian,
                              std::ostream* msgs = 0) {
        const double epsilon = 1e-3;
        const int order = 4;
        const double perturbations[order] 
          = {-2*epsilon, -1*epsilon, epsilon, 2*epsilon};
        const double coefficients[order]
          = { 1.0 / 12.0, 
              -2.0 / 3.0, 
              2.0 / 3.0, 
              -1.0 / 12.0 };

        double result 
          = model.template grad_log_prob<propto,
                                         jacobian_adjust_transform>(params_r, 
                                                                    params_i, 
                                                                    gradient, 
                                                                    msgs);
        hessian.assign(params_r.size() * params_r.size(), 0);
        std::vector<double> temp_grad(params_r.size());
        std::vector<double> perturbed_params(params_r.begin(), params_r.end());
        for (size_t d = 0; d < params_r.size(); d++) {
          double* row = &hessian[d*params_r.size()];
          for (int i = 0; i < order; i++) {
            perturbed_params[d] = params_r[d] + perturbations[i];
            model.template grad_log_prob<propto,jacobian_adjust_transform>(perturbed_params, params_i, temp_grad);
            for (size_t dd = 0; dd < params_r.size(); dd++) {
              row[dd] += 0.5 * coefficients[i] * temp_grad[dd] / epsilon;
              hessian[d + dd*params_r.size()] 
                += 0.5 * coefficients[i] * temp_grad[dd] / epsilon;
            }
          }
          perturbed_params[d] = params_r[d];
        }
        return result;
      }    

  }
}



#endif
