#ifndef STAN_SERVICES_UTIL_CREATE_RNG_HPP
#define STAN_SERVICES_UTIL_CREATE_RNG_HPP

#include <boost/random/mixmax.hpp>

namespace stan {

using rng_t = boost::random::mixmax;

namespace services {
namespace util {

/**
 * Creates a pseudo random number generator from a random seed
 * and a chain id by initializing the PRNG with the seed and
 * then advancing past pow(2, 50) times the chain ID draws to
 * ensure different chains sample from different segments of the
 * pseudo random number sequence.
 *
 * Chain IDs should be kept to larger values than one to ensure
 * that the draws used to initialized transformed data are not
 * duplicated.
 *
 * @param[in] seed the random seed
 * @param[in] chain the chain id
 * @return an stan::rng_t instance
 */
inline rng_t create_rng(unsigned int seed, unsigned int chain) {
  // RNG state is 128 bits, but user only provides 64 total bits
  // Additionally, there are issues if all 128 bits are 0, hence
  // the 1 as the second argument
  rng_t rng(0, 1, seed, chain);
  return rng;
}

}  // namespace util
}  // namespace services
}  // namespace stan
#endif
