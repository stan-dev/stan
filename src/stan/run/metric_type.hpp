#ifndef STAN_RUN_METRIC_TYPE_HPP
#define STAN_RUN_METRIC_TYPE_HPP

namespace stan {
namespace run {

enum class metric_t {
    UNIT_E = 0,
    DIAG_E = 1,
    DENSE_E = 2
};
}  // namespace run
}  // namespace stan
#endif
