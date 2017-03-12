#ifndef STAN_SERVICES_UTIL_EXPERIMENTAL_MESSAGE_HPP
#define STAN_SERVICES_UTIL_EXPERIMENTAL_MESSAGE_HPP

#include <stan/callbacks/writer.hpp>

namespace stan {
  namespace services {
    namespace util {

      /**
       * Writes an experimental message to the writer.
       * All experimental algorithms should call this function.
       *
       * @param[in,out] message_writer writer for experimental algorithm message
       */
      void experimental_message(stan::callbacks::writer&
                                message_writer) {
        message_writer("------------------------------"
                       "------------------------------");
        message_writer("EXPERIMENTAL ALGORITHM:");
        message_writer("  This procedure has not been thoroughly tested"
                       " and may be unstable");
        message_writer("  or buggy. The interface is subject to change.");
        message_writer("------------------------------"
                       "------------------------------");
        message_writer();
        message_writer();
      }

    }
  }
}

#endif
