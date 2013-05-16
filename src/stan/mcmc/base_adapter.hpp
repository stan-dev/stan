#ifndef __STAN__MCMC__BASE__ADAPTER__BETA__
#define __STAN__MCMC__BASE__ADAPTER__BETA__

namespace stan {
  
  namespace mcmc {
    
    class base_adapter {
      
    public:
      
      base_adapter(): _adapt_flag(false) {};
      
      virtual void engage_adaptation()    { _adapt_flag = true; }
      virtual void disengage_adaptation() { _adapt_flag = false; }
      
      bool adapting() { return _adapt_flag; }
      
    protected:
      
      bool _adapt_flag;
      
    };
    
  } // mcmc
  
} // stan

#endif
