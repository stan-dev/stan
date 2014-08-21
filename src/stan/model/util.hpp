#ifndef STAN__MODEL__UTIL_HPP
#define STAN__MODEL__UTIL_HPP

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/matrix/grad.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/autodiff.hpp>

namespace stan {

  namespace model {

    /**
     * Helper function to calculate log probability for 
     * <code>double</code> scalars up to a proportion.
     *
     * This implementation wraps the <code>double</code> values in
     * <code>stan::agrad::var</code> and calls the model's
     * <code>log_prob()</code> function with <code>propto=true</code>
     * and the specified parameter for applying the Jacobian
     * adjustment for transformed parameters.
     * 
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to 
     * the log probability.
     * @tparam M Class of model.
     * @param[in] model Model.
     * @param[in] params_r Real-valued parameters.
     * @param[in] params_i Integer-valued parameters.
     * @param[in,out] msgs
     */
    template <bool jacobian_adjust_transform, class M>
    double log_prob_propto(const M& model,
                           std::vector<double>& params_r,
                           std::vector<int>& params_i,
                           std::ostream* msgs = 0) {
      using stan::agrad::var;
      using std::vector;
      vector<var> ad_params_r;
      for (size_t i = 0; i < model.num_params_r(); ++i)
        ad_params_r.push_back(params_r[i]);
      try {
        double lp
          = model
          .template log_prob<true,
                             jacobian_adjust_transform>(ad_params_r,params_i,
                                                        msgs)
        .val();
        stan::agrad::recover_memory();
        return lp;
      } catch (std::exception &ex) {
        stan::agrad::recover_memory();
        throw;
      }
    }

    /**
     * Compute the gradient using reverse-mode automatic
     * differentiation, writing the result into the specified
     * gradient, using the specified perturbation.
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to 
     * the log probability.
     * @tparam M Class of model.
     * @param[in] model Model.
     * @param[in] params_r Real-valued parameters.
     * @param[in] params_i Integer-valued parameters.
     * @param[out] gradient Vector into which gradient is written.
     * @param[in,out] msgs
     */
    template <bool propto, bool jacobian_adjust_transform, class M>
    double log_prob_grad(const M& model,
                         std::vector<double>& params_r,
                         std::vector<int>& params_i,
                         std::vector<double>& gradient,
                         std::ostream* msgs = 0) {
      using std::vector;
      using stan::agrad::var;
      double lp;
      try {
        vector<var> ad_params_r(params_r.size());
        for (size_t i = 0; i < model.num_params_r(); ++i) {
          stan::agrad::var var_i(params_r[i]);
          ad_params_r[i] = var_i;
        }
        var adLogProb
          = model
          .template log_prob<propto,
                             jacobian_adjust_transform>(ad_params_r,
                                                        params_i,msgs);
        lp = adLogProb.val();
        adLogProb.grad(ad_params_r,gradient);
      } catch (const std::exception &ex) {
        stan::agrad::recover_memory();
        throw;
      }
      stan::agrad::recover_memory();
      return lp;
    }
    
    /**
     * Helper function to calculate log probability for
     * <code>double</code> scalars up to a proportion.
     *
     * This implementation wraps the <code>double</code> values in
     * <code>stan::agrad::var</code> and calls the model's
     * <code>log_prob()</code> function with <code>propto=true</code>
     * and the specified parameter for applying the Jacobian
     * adjustment for transformed parameters.
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to
     * the log probability.
     * @tparam M Class of model.
     * @param[in] model Model.
     * @param[in] params_r Real-valued parameters.
     * @param[in,out] msgs
     */
    template <bool jacobian_adjust_transform, class M>
    double log_prob_propto(const M& model,
                           Eigen::VectorXd& params_r,
                           std::ostream* msgs = 0) {
      using stan::agrad::var;
      using std::vector;
      double lp;
      try {
        vector<var> ad_params_r;
        for (size_t i = 0; i < model.num_params_r(); ++i)
          ad_params_r.push_back(params_r(i));
        lp
          = model
          .template log_prob<true,
                             jacobian_adjust_transform>(ad_params_r, msgs)
          .val();
      } catch (std::exception &ex) {
        stan::agrad::recover_memory();
        throw;
      }
      stan::agrad::recover_memory();
      return lp;
    }
    
    /**
     * Compute the gradient using reverse-mode automatic
     * differentiation, writing the result into the specified
     * gradient, using the specified perturbation.
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to
     * the log probability.
     * @tparam M Class of model.
     * @param[in] model Model.
     * @param[in] params_r Real-valued parameters.
     * @param[out] gradient Vector into which gradient is written.
     * @param[in,out] msgs
     */
    template <bool propto, bool jacobian_adjust_transform, class M>
    double log_prob_grad(const M& model,
                         Eigen::VectorXd& params_r,
                         Eigen::VectorXd& gradient,
                         std::ostream* msgs = 0) {
      using std::vector;
      using stan::agrad::var;
      Eigen::Matrix<var,Eigen::Dynamic,1> ad_params_r(params_r.size());
      for (size_t i = 0; i < model.num_params_r(); ++i) {
        stan::agrad::var var_i(params_r[i]);
        ad_params_r[i] = var_i;
      }
      try {
        var adLogProb
          = model
            .template log_prob<propto,
                               jacobian_adjust_transform>(ad_params_r, msgs);
        double val = adLogProb.val();
        stan::agrad::grad(adLogProb, ad_params_r, gradient);
        return val;
      } catch (std::exception &ex) {
        stan::agrad::recover_memory();
        throw;
      }
    }

    /**
     * Compute the gradient using finite differences for
     * the specified parameters, writing the result into the
     * specified gradient, using the specified perturbation.
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to the
     * log probability.
     * @tparam M Class of model.
     * @param model Model.
     * @param params_r Real-valued parameters.
     * @param params_i Integer-valued parameters.
     * @param[out] grad Vector into which gradient is written.
     * @param epsilon
     * @param[in,out] msgs
     */
    template <bool propto, bool jacobian_adjust_transform, class M>
    void finite_diff_grad(const M& model,
                          std::vector<double>& params_r,
                          std::vector<int>& params_i,
                          std::vector<double>& grad,
                          double epsilon = 1e-6,
                          std::ostream* msgs = 0) {
      std::vector<double> perturbed(params_r);
      grad.resize(params_r.size());
      for (size_t k = 0; k < params_r.size(); k++) {
        perturbed[k] += epsilon;
        double logp_plus 
          = model
          .template log_prob<propto,
                             jacobian_adjust_transform>(perturbed, params_i, 
                                                        msgs);
        perturbed[k] = params_r[k] - epsilon;
        double logp_minus
          = model
          .template log_prob<propto,
                             jacobian_adjust_transform>(perturbed, params_i,
                                                        msgs);
        double gradest = (logp_plus - logp_minus) / (2*epsilon);
        grad[k] = gradest;
        perturbed[k] = params_r[k]; 
      }
    }


    /**
     * Test the log_prob_grad() function's ability to produce
     * accurate gradients using finite differences.  This shouldn't
     * be necessary when using autodiff, but is useful for finding
     * bugs in hand-written code (or agrad).
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to the
     * log probability.
     * @tparam M Class of model.
     * @param model Model.
     * @param params_r Real-valued parameter vector.
     * @param params_i Integer-valued parameter vector.
     * @param epsilon Real-valued scalar saying how much to perturb. Defaults to 1e-6.
     * @param error Real-valued scalar saying how much error to allow. Defaults to 1e-6.
     * @param o Output stream for messages. Defaults to std::cout.
     * @param msgs Stream to which Stan programs write. Defaults to 0.
     * @return number of failed gradient comparisons versus allowed
     * error, so 0 if all gradients pass
     */
    template <bool propto, bool jacobian_adjust_transform, class M>
    int test_gradients(const M& model,
                       std::vector<double>& params_r,
                       std::vector<int>& params_i,
                       double epsilon = 1e-6,
                       double error = 1e-6,
                       std::ostream& o = std::cout,
                       std::ostream* msgs = 0) {
      std::vector<double> grad;
      double lp 
        = stan::model::log_prob_grad<propto,
                                     jacobian_adjust_transform>(model,
                                                                params_r,
                                                                params_i,
                                                                grad,msgs);
      
      std::vector<double> grad_fd;
      finite_diff_grad<false,
                       true,
                       M>(model,
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


    /**
     * Evaluate the log-probability, its gradient, and its Hessian
     * at params_r. This default version computes the Hessian
     * numerically by finite-differencing the gradient, at a cost of
     * O(params_r.size()^2).
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to the
     * log probability.
     * @tparam M Class of model.
     * @param model Model.
     * @param params_r Real-valued parameter vector.
     * @param params_i Integer-valued parameter vector.
     * @param gradient Vector to write gradient to.
     * @param hessian Vector to write gradient to. hessian[i*D + j]
     * gives the element at the ith row and jth column of the Hessian
     * (where D=params_r.size()).
     * @param msgs Stream to which print statements in Stan
     * programs are written, default is 0
     */
    template <bool propto, bool jacobian_adjust_transform, class M>
    double grad_hess_log_prob(const M& model,
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
        = stan::model::log_prob_grad<propto,
                                     jacobian_adjust_transform>(model,
                                                                params_r, 
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
          stan::model::log_prob_grad<propto,
                                     jacobian_adjust_transform>(model,
                                                            perturbed_params,
                                                            params_i, 
                                                            temp_grad);
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
    
    // Interface for automatic differentiation of models
    
    template <class M>
    struct model_functional {
      
      const M& model;
      std::ostream* o;
      
      model_functional(const M& m, std::ostream* out): model(m), o(out) {};
      
      template <typename T>
      T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
        return model.template log_prob<true, true, T>(x, o);
      }
      
    };
    
    template <class M>
    void gradient(const M& model,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                  double& f,
                  Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                  std::ostream* msgs = 0) {
      
      stan::agrad::gradient(model_functional<M>(model, msgs), x, f, grad_f);
      
    }
    
    template <class M>
    void hessian(const M& model,
                 const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                 double& f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& hess_f,
                 std::ostream* msgs = 0) {
      
      stan::agrad::hessian(model_functional<M>(model, msgs), x, f, grad_f, hess_f);
      
    }

    template <class M>
    void gradient_dot_vector(const M& model,
                             const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                             const Eigen::Matrix<double, Eigen::Dynamic, 1>& v,
                             double& f,
                             double& grad_f_dot_v,
                             std::ostream* msgs = 0) {
      
      stan::agrad::gradient_dot_vector(model_functional<M>(model, msgs), x, v, f, grad_f_dot_v);
      
    }
    
    template <class M>
    void hessian_times_vector(const M& model,
                              const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                              const Eigen::Matrix<double, Eigen::Dynamic, 1>& v,
                              double& f,
                              Eigen::Matrix<double, Eigen::Dynamic, 1>& hess_f_dot_v,
                              std::ostream* msgs = 0) {
      
      stan::agrad::hessian_times_vector(model_functional<M>(model, msgs), x, v, f, hess_f_dot_v);
      
    }
    
    template <class M>
    void grad_tr_mat_times_hessian(const M& model,
                                   const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                                   const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X,
                                   Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_tr_X_hess_f,
                                   std::ostream* msgs = 0) {
      
      stan::agrad::grad_tr_mat_times_hessian(model_functional<M>(model, msgs), x, X, grad_tr_X_hess_f);
      
    }
    

  }
}



#endif
