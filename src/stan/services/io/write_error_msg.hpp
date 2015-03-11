#ifndef STAN__SERVICES__IO__WRITE_ERROR_MSG_HPP
#define STAN__SERVICES__IO__WRITE_ERROR_MSG_HPP

#include <ostream>
#include <stdexcept>

namespace stan {
  namespace services {
    namespace io {
      
      template <class Writer>
      void write_error_msg(Writer& writer,
                           const std::exception& e) {
        writer.write_message("");
        writer.write_message("Informational Message: The current Metropolis "
                             + "proposal is about to be rejected because of the"
                             + "following issue:");
        writer.write_message(e.what());
        writer.write_message("If this warning occurs sporadically, such as for "
                             + "highly constrained variable types like covariance "
                             + "matrices, then the sampler is fine,");
        writer.write_message("but if this warning occurs often then your model "
                             + "may be either severely ill-conditioned or "
                             + "misspecified.");
      }
    
    }
  }
}

#endif
