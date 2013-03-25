#ifndef __STAN__MCMC__UTIL_HPP__
#define __STAN__MCMC__UTIL_HPP__

#include <cstddef>
#include <stdexcept>
#include <fstream>

#include <boost/exception/diagnostic_information.hpp> 
#include <boost/exception_ptr.hpp> 

namespace stan {

  namespace mcmc {

    void write_error_msgs(std::ostream* error_msgs,
                          const std::domain_error& e) {
      
      if (!error_msgs) return;
      
      *error_msgs << std::endl
                  << "Informational Message: The parameter state is about to be Metropolis"
                  << " rejected due to the following underlying, non-fatal (really)"
                  << " issue (and please ignore that what comes next might say 'error'): "
                  << e.what()
                  << std::endl
                  << "If the problem persists across multiple draws, you might have"
                  << " a problem with an initial state or a gradient somewhere."
                  << std::endl
                  << " If the problem does not persist, the resulting samples will still"
                  << " be drawn from the posterior."
                  << std::endl;
    
    }

  }

}

#endif
