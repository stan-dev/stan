#ifndef STAN_SERVICES_UTIL_EXPERIMENTAL_MESSAGE_HPP
#define STAN_SERVICES_UTIL_EXPERIMENTAL_MESSAGE_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>

namespace stan {
  namespace services {
    namespace util {

      void experimental_message(stan::interface_callbacks::writer::base_writer& message_writer) {
        message_writer("------------------------------------------------------------");
        message_writer("EXPERIMENTAL ALGORITHM:");
        message_writer("  - please expect frequent updates to the procedure");
        message_writer("  - please expect unexpected inference results");
        message_writer("------------------------------------------------------------");
        message_writer();
        message_writer();
      }
      
    }
  }
}

#endif
