#ifndef STAN__COMMON__CONTEXT_FACTORY_HPP
#define STAN__COMMON__CONTEXT_FACTORY_HPP

#include <fstream>
#include <stan/io/dump.hpp>

namespace stan {
  namespace common {

    class var_context_factory {
    public:
      var_context_factory(): context_(0) {} // C++11 Upgrade to nullptr
      ~var_context_factory() { if(context_) delete context_; }
      
      virtual bool create_var_context(std::string source) = 0;
      stan::io::var_context* var_context() { return context_; }
      
    protected:
      stan::io::var_context* context_;
    };

    // This should be defined in cmdstan
    class dump_factory: public var_context_factory {
    public:
      
      bool create_var_context(std::string source) {
        if (context_) delete context_;
        
        std::fstream source_stream(source.c_str(),
                                   std::fstream::in);
        if (source_stream.fail()) return false;
        
        context_ = new stan::io::dump(source_stream);
        source_stream.close();
        
        return true;
      }
      
    };
    
  }
}

#endif
