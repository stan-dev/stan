#ifndef STAN__INTERFACE_CALLBACKS__VAR_CONTEXT_FACTORY__VAR_CONTEXT_FACTORY_HPP
#define STAN__INTERFACE_CALLBACKS__VAR_CONTEXT_FACTORY__VAR_CONTEXT_FACTORY_HPP

#include <string>

namespace stan {
  namespace interface_callbacks {
    namespace var_context_factory {

      template <typename VARCON>
      class var_context_factory {
      public:
        var_context_factory() {}
        virtual VARCON operator()(const std::string source) = 0;
        typedef VARCON var_context_t;
      };
      
    }
  }
}

#endif
