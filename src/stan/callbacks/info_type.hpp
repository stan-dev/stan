#ifndef STAN_CALLBACKS_INFO_TYPE_HPP
#define STAN_CALLBACKS_INFO_TYPE_HPP

namespace stan {
namespace callbacks {

// see design_docs 0032-stan-output-formats

enum class table_info_type : int {
    DRAW_CONSTRAIN = 1,
    DRAW_ENGINE = 2,
    PARAMS_INITS = 3,
    PARAMS_GRADIENTS = 4,
    PARAMS_UNCNSTRN = 5
};

enum class struct_info_type : int {
    INV_METRIC = 1,
    RUN_CONFIG = 2,
    RUN_TIMING = 3,
    //  MODEL_VARS = constrained, unconstrained variables - name, type, size
};

enum class stream_info_type : int {
    TEXT_STREAM = 1,
    BINARY_STREAM = 2,
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
