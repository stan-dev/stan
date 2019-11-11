#ifndef STAN_MCMC_HMC_NUTS_SOFTABS_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_SOFTABS_NUTS_HPP

#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_metric.hpp>
#include <stan/mcmc/hmc/integrators/impl_leapfrog.hpp>

namespace stan {
namespace mcmc {
/**
 * The No-U-Turn sampler (NUTS) with multinomial sampling
 * with a Gaussian-Riemannian disintegration and SoftAbs metric
 */
template <class Model, class BaseRNG>
using softabs_nuts = base_nuts<Model, softabs_metric, impl_leapfrog, BaseRNG>;


}  // namespace mcmc
}  // namespace stan
#endif
