#ifndef STAN__MCMC__BASE__LEAPFROG__BETA
#define STAN__MCMC__BASE__LEAPFROG__BETA

#include <iostream>
#include <iomanip>
#include <stan/mcmc/hmc/integrators/base_integrator.hpp>

namespace stan {
  
  namespace mcmc {
    
    template <typename H, typename P>
    class base_leapfrog: public base_integrator<H, P> {
      
    public:
      
      base_leapfrog(std::ostream* o): base_integrator<H, P>(o) {};
      
      void evolve(P& z, H& hamiltonian, const double epsilon) {
        
        begin_update_p(z, hamiltonian, 0.5 * epsilon);
        
        update_q(z, hamiltonian, epsilon);
        hamiltonian.update(z);
        
        end_update_p(z, hamiltonian, 0.5 * epsilon);
        
      }
      
      void verbose_evolve(P& z, H& hamiltonian, const double epsilon) {
        
        this->out_stream_->precision(6);
        int width = 14;
        int nColumn = 4;
        
        if (this->out_stream_) {
        
          *(this->out_stream_) << "Verbose Hamiltonian Evolution, Step Size = " << epsilon << ":" << std::endl;
          *(this->out_stream_) << "    " << std::setw(nColumn * width) << std::setfill('-')
                               << "" << std::setfill(' ') << std::endl;
          *(this->out_stream_) << "    "
                               << std::setw(width) << std::left << "Poisson"
                               << std::setw(width) << std::left << "Initial"
                               << std::setw(width) << std::left << "Current"
                               << std::setw(width) << std::left << "DeltaH"
                               << std::endl;
          *(this->out_stream_) << "    "
                               << std::setw(width) << std::left << "Operator"
                               << std::setw(width) << std::left << "Hamiltonian"
                               << std::setw(width) << std::left << "Hamiltonian"
                               << std::setw(width) << std::left << "/ Stepsize^{2}"
                               << std::endl;
          *(this->out_stream_) << "    " << std::setw(nColumn * width) << std::setfill('-')
                               << "" << std::setfill(' ') << std::endl;
        
        }
          
        double H0 = hamiltonian.H(z);
        
        begin_update_p(z, hamiltonian, 0.5 * epsilon);
        
        double H1 = hamiltonian.H(z);
        
        if (this->out_stream_) {
        
          *(this->out_stream_) << "    "
                               << std::setw(width) << std::left << "hat{V}/2"
                               << std::setw(width) << std::left << H0
                               << std::setw(width) << std::left << H1
                               << std::setw(width) << std::left << (H1 - H0) / (epsilon * epsilon)
                               << std::endl;
          
        }
        
        update_q(z, hamiltonian, epsilon);
        hamiltonian.update(z);
        
        double H2 = hamiltonian.H(z);
        
        if (this->out_stream_) {
        
          *(this->out_stream_) << "    "
                               << std::setw(width) << std::left << "hat{T}"
                               << std::setw(width) << std::left << H0
                               << std::setw(width) << std::left << H2
                               << std::setw(width) << std::left << (H2 - H0) / (epsilon * epsilon)
                               << std::endl;
          
        }
        
        end_update_p(z, hamiltonian, 0.5 * epsilon);
        
        double H3 = hamiltonian.H(z);
        
        if (this->out_stream_) {
        
          *(this->out_stream_) << "    "
                               << std::setw(width) << std::left << "hat{V}/2"
                               << std::setw(width) << std::left << H0
                               << std::setw(width) << std::left << H3
                               << std::setw(width) << std::left << (H3 - H0) / (epsilon * epsilon)
                               << std::endl;
          
          *(this->out_stream_) << "    " << std::setw(nColumn * width) << std::setfill('-')
                               << "" << std::setfill(' ') << std::endl;
          
        }
        
      }
      
      virtual void begin_update_p(P& z, H& hamiltonian, double epsilon) = 0;
      virtual void update_q(P& z, H& hamiltonian, double epsilon) = 0;
      virtual void end_update_p(P& z, H& hamiltonian, double epsilon) = 0;
      
    };
    
  } // mcmc
  
} // stan


#endif
