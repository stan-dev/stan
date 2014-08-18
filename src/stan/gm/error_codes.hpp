#ifndef STAN__GM__ERROR_CODES_HPP
#define STAN__GM__ERROR_CODES_HPP

namespace stan {
  namespace gm {

    struct error_codes {
      // defining error codes to follow FreeBSD sysexits conventions
      // http://www.gsp.com/cgi-bin/man.cgi?section=3&topic=sysexits
      enum {
        OK = 0,
        USAGE = 64,
        DATAERR = 65,
        NOINPUT = 66,
        SOFTWARE = 70,
        CONFIG = 78
      };
    };
  }
}
#endif
