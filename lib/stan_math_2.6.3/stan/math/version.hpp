#ifndef STAN_MATH_VERSION_HPP
#define STAN_MATH_VERSION_HPP

#include <string>

#ifndef STRING_EXPAND
#define STRING_EXPAND(s) #s
#endif

#ifndef STRING
#define STRING(s) STRING_EXPAND(s)
#endif

#define STAN_MATH_MAJOR 2
#define STAN_MATH_MINOR 6
#define STAN_MATH_PATCH 3

namespace stan {
  namespace math {

    /** Major version number for Stan math library. */
    const std::string MAJOR_VERSION = STRING(STAN_MATH_MAJOR);

    /** Minor version number for Stan math library. */
    const std::string MINOR_VERSION = STRING(STAN_MATH_MINOR);

    /** Patch version for Stan math library. */
    const std::string PATCH_VERSION = STRING(STAN_MATH_PATCH);

  }
}

#endif
