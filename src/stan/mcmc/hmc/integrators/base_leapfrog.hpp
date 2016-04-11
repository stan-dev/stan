#ifndef STAN_MCMC_HMC_INTEGRATORS_BASE_LEAPFROG_HPP
#define STAN_MCMC_HMC_INTEGRATORS_BASE_LEAPFROG_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/hmc/integrators/base_integrator.hpp>
#include <iostream>
#include <iomanip>

namespace stan {
  namespace mcmc {

    template <class Hamiltonian>
    class base_leapfrog : public base_integrator<Hamiltonian> {
    public:
      base_leapfrog()
        : base_integrator<Hamiltonian>() {}

      void evolve(typename Hamiltonian::PointType& z,
                  Hamiltonian& hamiltonian,
                  const double epsilon,
                  interface_callbacks::writer::base_writer& writer) {
        begin_update_p(z, hamiltonian, 0.5 * epsilon, writer);

        update_q(z, hamiltonian, epsilon, writer);
        hamiltonian.update(z, writer); // Remove this and let implementations
                                       // be responsible for updating gradient?

        end_update_p(z, hamiltonian, 0.5 * epsilon, writer);
      }

      void verbose_evolve(typename Hamiltonian::PointType& z,
                          Hamiltonian& hamiltonian,
                          const double epsilon,
                          interface_callbacks::writer::base_writer& writer) {
        std::stringstream msg;
        msg.precision(6);

        int width = 14;
        int nColumn = 4;

        msg << "Verbose Hamiltonian Evolution, Step Size = " << epsilon << ":";
        writer(msg.str());

        msg.str("");
        msg << "    " << std::setw(nColumn * width)
            << std::setfill('-')
            << "" << std::setfill(' ');
        writer(msg.str());

        msg.str("");
        msg << "    "
            << std::setw(width) << std::left << "Poisson"
            << std::setw(width) << std::left << "Initial"
            << std::setw(width) << std::left << "Current"
            << std::setw(width) << std::left << "DeltaH";
        writer(msg.str());

        msg.str("");
        msg << "    "
            << std::setw(width) << std::left << "Operator"
            << std::setw(width) << std::left << "Hamiltonian"
            << std::setw(width) << std::left << "Hamiltonian"
            << std::setw(width) << std::left << "/ Stepsize^{2}";
        writer(msg.str());

        msg.str("");
        msg << "    " << std::setw(nColumn * width)
            << std::setfill('-')
            << "" << std::setfill(' ');
        writer(msg.str());

        double H0 = hamiltonian.H(z);

        begin_update_p(z, hamiltonian, 0.5 * epsilon, writer);

        double H1 = hamiltonian.H(z);

        msg.str("");
        msg << "    "
            << std::setw(width) << std::left << "hat{V}/2"
            << std::setw(width) << std::left << H0
            << std::setw(width) << std::left << H1
            << std::setw(width) << std::left << (H1 - H0) / (epsilon * epsilon);
        writer(msg.str());

        update_q(z, hamiltonian, epsilon, writer);
        hamiltonian.update(z, writer);

        double H2 = hamiltonian.H(z);

        msg.str("");
        msg << "    "
            << std::setw(width) << std::left << "hat{T}"
            << std::setw(width) << std::left << H0
            << std::setw(width) << std::left << H2
            << std::setw(width) << std::left << (H2 - H0) / (epsilon * epsilon);
        writer(msg.str());

        end_update_p(z, hamiltonian, 0.5 * epsilon, writer);

        double H3 = hamiltonian.H(z);

        msg.str("");
        msg << "    "
            << std::setw(width) << std::left << "hat{V}/2"
            << std::setw(width) << std::left << H0
            << std::setw(width) << std::left << H3
            << std::setw(width) << std::left << (H3 - H0) / (epsilon * epsilon);
        writer(msg.str());

        msg.str("");
        msg << "    "
            << std::setw(nColumn * width)
            << std::setfill('-')
            << ""
            << std::setfill(' ');
        writer(msg.str());
      }

      virtual
      void begin_update_p(typename Hamiltonian::PointType& z,
                          Hamiltonian& hamiltonian, double epsilon,
                          interface_callbacks::writer::base_writer& writer) = 0;
      virtual
      void update_q(typename Hamiltonian::PointType& z,
                    Hamiltonian& hamiltonian, double epsilon,
                    interface_callbacks::writer::base_writer& writer) = 0;
      virtual
      void end_update_p(typename Hamiltonian::PointType& z,
                        Hamiltonian& hamiltonian, double epsilon,
                        interface_callbacks::writer::base_writer& writer) = 0;
    };

  }  // mcmc
}  // stan
#endif
