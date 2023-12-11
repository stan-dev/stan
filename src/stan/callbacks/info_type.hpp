#ifndef STAN_CALLBACKS_INFO_TYPE_HPP
#define STAN_CALLBACKS_INFO_TYPE_HPP

namespace stan {
namespace callbacks {

enum class info_type : int {
  // see design_docs 0032-stan-output-formats
    LOG_PROB = 1,
    ALGORITHM_STATE = 2,
    DRAW_CONSTRAINED = 3,
    DRAW_UNCONSTRAINED = 4,
    GRADIENTS = 5,
    METRIC = 6,
    MODEL_METADATA = 7,
    CONFIG = 8,
    TIMING = 9
};

}  // namespace callbacks
}  // namespace stan
#endif
