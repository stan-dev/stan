#ifndef STAN_SERVICES_UTIL_RNG_HPP
#define STAN_SERVICES_UTIL_RNG_HPP

#include <boost/random/additive_combine.hpp>

namespace stan {
  namespace services {
    namespace util {

      boost::ecuyer1988 rng(unsigned int random_seed, unsigned int chain) {
        boost::ecuyer1988 rng(random_seed);
        
        // Advance generator to avoid process conflicts
        static boost::uintmax_t DISCARD_STRIDE
          = static_cast<boost::uintmax_t>(1) << 50;
        rng.discard(DISCARD_STRIDE * (chain - 1));

        return rng;
      }
      
    }
  }
}

#endif
