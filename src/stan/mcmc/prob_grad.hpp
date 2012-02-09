#ifndef __STAN__MCMC__PROB_GRAD_HPP__
#define __STAN__MCMC__PROB_GRAD_HPP__

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <stdio.h>

#include <stan/io/csv_writer.hpp>

namespace stan {

  namespace mcmc {

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
                                   std::vector<double>& gradient) = 0;

      virtual double log_prob(std::vector<double>& params_r, 
                              std::vector<int>& params_i) = 0;

      virtual double log_prob_star(size_t idx, 
                                   int val,
                                   std::vector<double>& params_r,
                                   std::vector<int>& params_i) {
        if (idx >= num_params_i()) // || idx < 0
          throw std::runtime_error ("idx >= num_params_i()");
        if (val >= param_range_i(idx).first) // 
          throw std::runtime_error ("val <= param_range_i(idx) lower");
        if (val >= param_range_i(idx).second) //
          throw std::runtime_error ("val >= param_range_i(idx) upper");

        int original_val = params_i[idx];
        params_i[idx] = val;
        double result = log_prob(params_r,params_i);
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
       * @param o Output stream to which values are written
       */
      virtual void write_csv(std::vector<double>& params_r,
                             std::vector<int>& params_i,
                             std::ostream& o) {
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
       * @params grad Vector into which gradient is written.
       */
      void finite_diff_grad(std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& grad,
                            double epsilon = 1e-6) {
        std::vector<double> perturbed(params_r);
        double logp = grad_log_prob(params_r, params_i, grad);
        grad.resize(params_r.size());
        for (size_t k = 0; k < params_r.size(); k++) {
          perturbed[k] += epsilon;
          double logp2 = log_prob(perturbed, params_i);
          double gradest = (logp2 - logp) / epsilon;
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
       * params_r. Defaults to 1e-6.
       */
      void test_gradients(std::vector<double>& params_r,
                          std::vector<int>& params_i,
                          double epsilon = 1e-6,
                          std::ostream& o = std::cout) {
        std::vector<double> grad;
        grad_log_prob(params_r,params_i,grad);

        std::vector<double> grad_fd;
        finite_diff_grad(params_r,params_i,grad_fd,epsilon);

        o << std::endl
          << std::setw(8) << "param k"
          << std::setw(16) << "model"
          << std::setw(16) << "finite diff"
          << std::setw(16) << "error" 
          << std::endl;
        for (size_t k = 0; k < params_r.size(); k++) {
          o << std::setw(8) << k
            << std::setw(16) << grad[k]
            << std::setw(16) << grad_fd[k]
            << std::setw(16) << (grad[k] - grad_fd[k])
            << std::endl;
        }
      }

      /**
       * Test gradients computed by finite differences against those
       * computed by the model, using random initialization for
       * real and integer parameters.
       *
       * @param epsilon Finite d
       */
      void test_gradients_random(double epsilon = 1e-6,
                                 std::ostream& o = std::cout) {
        std::vector<double> params_r(num_params_r());
        std::vector<int> params_i(num_params_i());
        for (size_t i = 0; i < params_r.size(); ++i)
          params_r[i] = 0.0; // FIXME:  randomize
        for (size_t i = 0; i < params_i.size(); ++i)
          params_i[i] = 0;   // FIXME:  randomize, keep in range
        test_gradients(params_r,params_i,epsilon,o);
      }
                                 

    };
  }
}

#endif
