#ifndef STAN_MODEL_PROB_GRAD_HPP
#define STAN_MODEL_PROB_GRAD_HPP

#include <cstddef>
#include <utility>
#include <vector>

namespace stan {

  namespace model {

    /**
     * The <code>prob_grad</code> class represents the basic parameter
     * holders for a model.  The command <code>bin/stanc</code> builds
     * Models extending this base helper class.
     */
    class prob_grad {
    protected:
      size_t num_params_r_;
      std::vector<std::pair<int, int> > param_ranges_i_;

    public:
      explicit prob_grad(size_t num_params_r)
        : num_params_r_(num_params_r),
          param_ranges_i_(std::vector<std::pair<int, int> >(0)) {
      }

      prob_grad(size_t num_params_r,
                std::vector<std::pair<int, int> >& param_ranges_i)
        : num_params_r_(num_params_r),
          param_ranges_i_(param_ranges_i) {
      }

      virtual ~prob_grad() { }

      inline size_t num_params_r() const {
        return num_params_r_;
      }

      inline size_t num_params_i() const {
        return param_ranges_i_.size();
      }

      inline std::pair<int, int> param_range_i(size_t idx) const {
        return param_ranges_i_[idx];
      }
    };
  }
}

#endif
