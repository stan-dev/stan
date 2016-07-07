#ifndef STAN_SERVICES_IO_WRITE_ERROR_MSG_HPP
#define STAN_SERVICES_IO_WRITE_ERROR_MSG_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <ostream>
#include <stdexcept>

namespace stan {
  namespace services {
    namespace io {

      /**
       * Writes a Metropolis rejection message.
       *
       * @param writer Writer callback
       * @param e Input exception
       */
      void write_error_msg(interface_callbacks::writer::base_writer& writer,
                           const std::exception& e) {
        writer("Informational Message: The current Metropolis"
               " proposal is about to be rejected because of"
               " the following issue:");
        writer(e.what());
        writer("If this warning occurs sporadically, such as"
               " for highly constrained variable types like"
               " covariance matrices, then the sampler is fine,");
        writer("but if this warning occurs often then your model"
               " may be either severely ill-conditioned or"
               " misspecified.");
        writer();
      }

    }
  }
}

#endif
