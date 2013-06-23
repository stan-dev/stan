#ifndef __STAN__MODEL__PROB_GRAD_HPP__
#define __STAN__MODEL__PROB_GRAD_HPP__

#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <stdio.h>
#include <utility>
#include <vector>

#include <stan/io/csv_writer.hpp>

namespace stan {

  namespace model {

    /**
     * The <code>prob_grad</code> class represents densities with
     * fixed numbers of discrete and scalar parameters and the
     * gradient with respet to the scalar parameters.
     *
     */
    class prob_grad {
    protected:
      size_t num_params_r__;
      std::vector<std::pair<int,int> > param_ranges_i__;

    public:

      prob_grad(size_t num_params_r)
        : num_params_r__(num_params_r),
          param_ranges_i__(std::vector<std::pair<int,int> >(0)) {
      }

      prob_grad(size_t num_params_r,
                std::vector<std::pair<int,int> >& param_ranges_i)
        : num_params_r__(num_params_r),
          param_ranges_i__(param_ranges_i) { 
      }

      virtual ~prob_grad() { }

      void set_num_params_r(size_t num_params_r) {
        num_params_r__ = num_params_r;
      }

      void setparam_ranges_i__(std::vector<std::pair<int,int> > param_ranges_i) {
        param_ranges_i__ = param_ranges_i;
      }

      virtual size_t num_params_r() {
        return num_params_r__;
      }

      virtual size_t num_params_i() {
        return param_ranges_i__.size();
      }

      inline std::pair<int,int> param_range_i(size_t idx) {
        return param_ranges_i__[idx];
      }

      inline void set_param_range_i_lower(size_t idx, int low) {
        param_ranges_i__[idx].first = low;
      }

      inline void set_param_range_i_upper(size_t idx, int up) {
        param_ranges_i__[idx].second = up;
      }

      inline int param_range_i_lower(size_t idx) {
        return param_ranges_i__[idx].first;
      }

      inline int param_range_i_upper(size_t idx) {
        return param_ranges_i__[idx].second;
      }

      virtual void init(std::vector<double>& params_r, 
                        std::vector<int>& params_i) {
        for (size_t i = 0; i < num_params_r(); ++i)
          params_r[i] = 0.0;
        for (size_t j = 0; j < num_params_i(); ++j)
          params_i[j] = param_range_i_lower(j);
      }

      virtual double grad_log_prob(std::vector<double>& params_r, 
                                   std::vector<int>& params_i,
                                   std::vector<double>& gradient,
                                   std::ostream* output_stream = 0) = 0;

      virtual double log_prob(std::vector<double>& params_r, 
                              std::vector<int>& params_i,
                              std::ostream* output_stream = 0) = 0;

      /**
       * Evaluate the log-probability, its gradient, and its Hessian
       * at params_r. This default version computes the Hessian
       * numerically by finite-differencing the gradient, at a cost of
       * O(params_r.size()^2).
       *
       * @param params_r Real-valued parameter vector.
       * @param params_i Integer-valued parameter vector.
       * @param gradient Vector to write gradient to.
       * @param hessian Vector to write gradient to. hessian[i*D + j]
       * gives the element at the ith row and jth column of the Hessian
       * (where D=params_r.size()).
       * @param output_stream Stream to which print statements in Stan
       * programs are written, default is 0
       */
      virtual double grad_hess_log_prob(std::vector<double>& params_r, 
                                        std::vector<int>& params_i,
                                        std::vector<double>& gradient,
                                        std::vector<double>& hessian,
                                        std::ostream* output_stream = 0) {
        const double epsilon = 1e-3;
        const int order = 4;
        const double perturbations[order] = {-2*epsilon, -1*epsilon, epsilon, 2*epsilon};
        const double coefficients[order] = {1.0/12.0,-2.0/3.0,2.0/3.0,-1.0/12.0};

        double result = grad_log_prob(params_r, params_i, gradient, 
                                      output_stream);

        hessian.assign(params_r.size() * params_r.size(), 0);
        std::vector<double> temp_grad(params_r.size());
        std::vector<double> perturbed_params(params_r.begin(), params_r.end());
        for (size_t d = 0; d < params_r.size(); d++) {
          double* row = &hessian[d*params_r.size()];
          for (int i = 0; i < order; i++) {
            perturbed_params[d] = params_r[d] + perturbations[i];
            grad_log_prob(perturbed_params, params_i, temp_grad);
            for (size_t dd = 0; dd < params_r.size(); dd++) {
              row[dd] += 0.5 * coefficients[i] * temp_grad[dd] / epsilon;
              hessian[d + dd*params_r.size()] += 0.5 * coefficients[i] * temp_grad[dd] / epsilon;
            }
          }
          perturbed_params[d] = params_r[d];
        }

        return result;
      }

      virtual double log_prob_star(size_t idx, 
                                   int val,
                                   std::vector<double>& params_r,
                                   std::vector<int>& params_i,
                                   std::ostream* output_stream = 0) {
        if (idx >= num_params_i()) // || idx < 0
          throw std::runtime_error ("idx >= num_params_i()");
        if (val >= param_range_i(idx).first) // 
          throw std::runtime_error ("val <= param_range_i(idx) lower");
        if (val >= param_range_i(idx).second) //
          throw std::runtime_error ("val >= param_range_i(idx) upper");

        int original_val = params_i[idx];
        params_i[idx] = val;
        double result = log_prob(params_r,params_i,output_stream);
        params_i[idx] = original_val;
        return result;
      }


      /**
       * Write the parameters on a single line in CSV format.  The implementation in
       * this abstract base class writes out the free parameters as
       * viwed by HMC.  Subclasses may extend this class and override
       * this implementation to return constrained values which may differ
       * in both number and scale from the raw unconstrained parameters.
       *
       * @param params_r Real-valued parameter vector.
       * @param params_i Integer-valued parameter vector.
       * @params o Stream to which CSV file is written
       * @param output_stream Stream to which print statements in Stan
       * programs are written, default is 0
       */
      template <typename RNG>
      void write_csv(std::vector<double>& params_r,
                     std::vector<int>& params_i,
                     std::ostream& o,
                     RNG& /* rng */,
                     std::ostream* /*output_stream = 0*/) {
        stan::io::csv_writer writer(o);
        for (size_t i = 0; i < params_i.size(); ++i)
          writer.write(params_i[i]);
        for (size_t i = 0; i < params_r.size(); ++i)
          writer.write(params_r[i]);
        writer.newline();
      }


      /**
       * Compute the gradient using finite differences for
       * the specified parameters, writing the result into the
       * specified gradient, using the specified perturbation.
       *
       * @param params_r Real-valued parameters.
       * @param params_i Integer-valued parameters.
       * @param[out] grad Vector into which gradient is written.
       * @param epsilon
       * @param[in,out] output_stream
       */
      void finite_diff_grad(std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& grad,
                            double epsilon = 1e-6,
                            std::ostream* output_stream = 0) {
        std::vector<double> perturbed(params_r);
        grad_log_prob(params_r,params_i,grad,output_stream);
        grad.resize(params_r.size());
        for (size_t k = 0; k < params_r.size(); k++) {
          perturbed[k] += epsilon;
          double logp_plus = log_prob(perturbed,params_i,output_stream);
          perturbed[k] = params_r[k] - epsilon;
          double logp_minus = log_prob(perturbed,params_i,output_stream);
          double gradest = (logp_plus - logp_minus) / (2*epsilon);
          grad[k] = gradest;
          perturbed[k] = params_r[k]; 
        }
      }

      /**
       * Test the grad_log_prob() function's ability to produce
       * accurate gradients using finite differences.  This shouldn't
       * be necessary when using autodiff, but is useful for finding
       * bugs in hand-written code (or agrad).
       *
       * @param params_r Real-valued parameter vector.
       * @param params_i Integer-valued parameter vector.
       * @param epsilon Real-valued scalar saying how much to perturb 
       * @param error Real-valued scalar saying how much error to allow
       * @param o Output stream for messages.
       * params_r. Defaults to 1e-6.
       * @param output_stream Stream to which Stan programs write.
       * @return number of failed gradient comparisons versus allowed
       * error, so 0 if all gradients pass
       */
      int test_gradients(std::vector<double>& params_r,
                         std::vector<int>& params_i,
                         double epsilon = 1e-6,
                         double error = 1e-6,
                         std::ostream& o = std::cout,
                         std::ostream* output_stream = 0) {
        std::vector<double> grad;
        double lp = grad_log_prob(params_r,params_i,grad,output_stream);
        
        std::vector<double> grad_fd;
        finite_diff_grad(params_r,params_i,grad_fd,epsilon,output_stream);

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

    };
  }
}

#endif
