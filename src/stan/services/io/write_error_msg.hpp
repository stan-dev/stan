#ifndef STAN_SERVICES_IO_WRITE_ERROR_MSG_HPP
#define STAN_SERVICES_IO_WRITE_ERROR_MSG_HPP

#include <ostream>
#include <stdexcept>

namespace stan {
  namespace services {
    namespace io {
      
      template <class Writer>
      void write_error_msg(Writer& writer,
                           const std::exception& e) {
        writer();
        writer("Informational Message: The current Metropolis "
               "proposal is about to be rejected because of the "
               "following issue:");
        writer(e.what());
        writer("If this warning occurs sporadically, such as for "
               "highly constrained variable types like covariance "
               "matrices, then the sampler is fine,"
               "but if this warning occurs often then your model "
               "may be either severely ill-conditioned or "
               "misspecified.");
      }
    
    }
  }
}

#endif
