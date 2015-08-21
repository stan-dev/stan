#ifndef STAN_INTERFACE_CALLBACKS_WRITER_STREAM_WRITER_TYPEDEFS_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_STREAM_WRITER_TYPEDEFS_HPP

#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <sstream>
#include <iostream>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      // An implementation of stream_writer
      // using stringstream for unit tests
      typedef stream_writer<std::stringstream> sstream_writer;
    }
  }
}

#endif
