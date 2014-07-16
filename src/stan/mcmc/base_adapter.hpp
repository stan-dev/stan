#ifndef __STAN__MCMC__BASE__ADAPTER__BETA__
#define __STAN__MCMC__BASE__ADAPTER__BETA__

namespace stan {
  
  namespace mcmc {
    
    class base_adapter {
      
    public:
      
      base_adapter(): adapt_flag_(false) {};
      
      virtual void engage_adaptation()    { adapt_flag_ = true; }
      virtual void disengage_adaptation() { adapt_flag_ = false; }
      
      bool adapting() { return adapt_flag_; }
      
    protected:
      
      bool adapt_flag_;
      
    };
    
  } // mcmc
  
} // stan

#endif
