#ifndef STAN_RUN_ALGORITHM_TYPE_HPP
#define STAN_RUN_ALGORITHM_TYPE_HPP

namespace stan {
namespace run {

enum class algorithm_t {
    STAN2_HMC = 1,
    MLE = 2,
    PATHFINDER = 3,
    ADVI = 4,
    STANDALONE_GQ = 5
};

}  // namespace run
}  // namespace stan
#endif
