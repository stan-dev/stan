#ifndef __STAN__IO__UTIL_HPP__
#define __STAN__IO__UTIL_HPP__

#include <string>
#include <ctime>

namespace stan {

  namespace io {

    /**
     * Return the current coordinated universal time (UTC) as a string.
     * 
     * Output is of the form "Fri Feb 24 21:15:36 2012"
     *
     * @return String representation of current UTC.
     */
    std::string utc_time_string() {
      std::time_t rawtime = time(0);
      std::tm *time = gmtime(&rawtime);
      return std::string(asctime(time));
    }

  }
}

#endif
