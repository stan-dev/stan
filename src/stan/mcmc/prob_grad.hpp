#ifndef __STAN__MCMC__PROB_GRAD_HPP__
#define __STAN__MCMC__PROB_GRAD_HPP__

#include <stdexcept>
#include <vector>
#include <limits>
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
      unsigned int num_params_r__;
      std::vector<int> param_ranges_i__;

    public:

      prob_grad(unsigned int num_params_r)
        : num_params_r__(num_params_r),
          param_ranges_i__(std::vector<int>(0)) {
      }

      prob_grad(unsigned int num_params_r,
                std::vector<int>& param_ranges_i)
        : num_params_r__(num_params_r),
          param_ranges_i__(param_ranges_i) { 
      }

      virtual ~prob_grad() { }

      void set_num_params_r(unsigned int num_params_r) {
        num_params_r__ = num_params_r;
      }

      void setparam_ranges_i__(std::vector<int> param_ranges_i) {
        param_ranges_i__ = param_ranges_i;
      }

      virtual unsigned int num_params_r() {
        return num_params_r__;
      }

      virtual unsigned int num_params_i() {
        return param_ranges_i__.size();
      }

      inline int param_range_i(unsigned int idx) {
        return param_ranges_i__[idx];
      }

      virtual void init(std::vector<double>& params_r, 
                        std::vector<int>& params_i) {
        for (unsigned int i = 0; i < num_params_r(); ++i)
          params_r[i] = 0.0;
        for (unsigned int j = 0; j < num_params_i(); ++j)
          params_i[j] = 0;
      }

      virtual double grad_log_prob(std::vector<double>& params_r, 
                                   std::vector<int>& params_i,
                                   std::vector<double>& gradient) = 0;

      virtual double log_prob(std::vector<double>& params_r, 
                              std::vector<int>& params_i) = 0;

      virtual double log_prob_star(unsigned int idx, 
                                   int val,
                                   std::vector<double>& params_r,
                                   std::vector<int>& params_i) {
        if (idx >= num_params_i()) // || idx < 0
          throw std::runtime_error ("idx >= num_params_i()");
        if (val >= param_range_i(idx)) // || val < 0
          throw std::runtime_error ("val >= param_range_i(idx)");

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
        for (unsigned int i = 0; i < params_i.size(); ++i)
          writer.write(params_i[i]);
        for (unsigned int i = 0; i < params_r.size(); ++i)
          writer.write(params_r[i]);
        writer.newline();
      }
    };
  }
}

#endif
