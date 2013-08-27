#ifndef __STAN__DIFF__REV__CALCULATE_CHAIN_HPP__
#define __STAN__DIFF__REV__CALCULATE_CHAIN_HPP__

#include <valarray>

namespace stan {
  namespace diff {
      inline double calculate_chain(const double& x, const double& val) {
        return std::exp(x - val); // works out to inv_logit(x)
      }
  }
}
#endif
