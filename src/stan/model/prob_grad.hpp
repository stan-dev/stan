#ifndef __STAN__MODEL__PROB_GRAD_HPP__
#define __STAN__MODEL__PROB_GRAD_HPP__

#include <utility>
#include <vector>

#include <stan/io/csv_writer.hpp>

namespace stan {

  namespace model {

    /**
     * The <code>prob_grad</code> class represents the basic parameter
     * holders for a model.  The command <code>bin/stanc</code> builds
     * Models extending this base helper class.
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

      inline size_t num_params_r() const {
        return num_params_r__;
      }

      inline size_t num_params_i() const {
        return param_ranges_i__.size();
      }

      inline std::pair<int,int> param_range_i(size_t idx) const {
        return param_ranges_i__[idx];
      }

    };
  }
}

#endif
