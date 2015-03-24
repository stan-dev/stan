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
        writer();
        writer(std::string("Informational Message: The current Metropolis ")
               + std::string("proposal is about to be rejected because of the")
               + std::string("following issue:"));
        writer(e.what());
        writer(std::string("If this warning occurs sporadically, such as for ")
               + std::string("highly constrained variable types like covariance ")
               + std::string("matrices, then the sampler is fine,"));
        writer(std::string("but if this warning occurs often then your model ")
               + std::string("may be either severely ill-conditioned or ")
               + std::string("misspecified."));
      }
    
    }
  }
}

#endif
