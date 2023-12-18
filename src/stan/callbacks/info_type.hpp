#ifndef STAN_CALLBACKS_INFO_TYPE_HPP
#define STAN_CALLBACKS_INFO_TYPE_HPP

namespace stan {
namespace callbacks {

enum class info_type : int {
  // see design_docs 0032-stan-output-formats
    DRAW_CONSTRAINED = 1,
    ENGINE_STATE = 2,
    LOG_PROB = 3,
    METRIC = 4,
    MODEL_METADATA = 5,
    PARAMS_UNCONSTRAINED = 6,
    PARAMS_GRADIENTS = 7,
    RUN_CONFIG = 8,
    RUN_TIMING = 9,
    VALID_INIT_PARAMS = 10,
};

}  // namespace callbacks
}  // namespace stan
#endif
