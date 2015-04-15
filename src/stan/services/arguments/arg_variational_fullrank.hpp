#ifndef __STAN__SERVICES__ARGUMENTS__VARIATIONAL__FULLRANK__HPP__
#define __STAN__SERVICES__ARGUMENTS__VARIATIONAL__FULLRANK__HPP__

#include <stan/services/arguments/categorical_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_fullrank: public categorical_argument {
    public:
      arg_variational_fullrank() {
        _name = "fullrank";
        _description = "full-rank covariance";
      }
    };
  }  // services
}  // stan

#endif

