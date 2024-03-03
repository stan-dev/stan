#ifndef STAN_CALLBACKS_INFO_TYPE_HPP
#define STAN_CALLBACKS_INFO_TYPE_HPP

namespace stan {
namespace callbacks {

// see design_docs 0032-stan-output-formats

// table writers:  CSV or Apache arrow
enum class table_info_type : int {
  // HMC outputs
  ALGO_STATE = 1,
  DRAW_SAMPLE = 2,
  DRAW_WARMUP = 3,
  PARAMS_INIT = 4,
  UPARAMS_WARMUP = 5,  // method diagnostics
  UPARAMS_SAMPLE = 6,  // method diagnostics
};

// struct writers: JSON
enum class struct_info_type : int {
    RUN_CONFIG = 1,
    RUN_TIMING = 2,
    MODEL_VARS = 3,  // constrained, unconstrained variables - name, type, size
    // HMC outputs
    INV_METRIC = 4,
    // other outputs ?
    TEST_GRADIENTS = 5,   // method diagnose: log_prob_grad, model::finite_diff_grad
    LOG_PROB_GRAD = 6  // method log_prob
};

// stream writers:  text, raw
enum class stream_info_type : int {
    TEXT_STREAM = 1,
    BINARY_STREAM = 2
};

template<typename T>
struct is_info_type_enum : std::false_type {};

template<>
struct is_info_type_enum<table_info_type> : std::true_type {};

template<>
struct is_info_type_enum<struct_info_type> : std::true_type {};

template<>
struct is_info_type_enum<stream_info_type> : std::true_type {};


}  // namespace callbacks
}  // namespace stan
#endif
