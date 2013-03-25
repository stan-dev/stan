#ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__


#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <fstream>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>
#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/mcmc/static_hmc.hpp>

namespace stan {

  namespace gm {

        
    void print_nuts_help(std::string cmd) {
      using stan::io::print_help_option;

      std::cout << std::endl;
      std::cout << "STAN I AM IN YOU" << std::endl;
      std::cout << std::endl << std::endl;

    }

    bool do_print(int n, int refresh) {
      return (refresh > 0)
        && (n == 0
            || ((n + 1) % refresh == 0) );
    }

    template <class Model>
    int nuts_command(int argc, const char* argv[]) {

      stan::io::cmd_line command(argc,argv);

      if (command.has_flag("help")) {
        print_nuts_help(argv[0]);
        return 0;
      }

      std::string data_file;
      command.val("data",data_file);
      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();

      Model model(data_var_context, &std::cout);
      //stan::model::prob_grad model(data_var_context, &std::cout);
      
      typedef boost::ecuyer1988 rng_t;
      
      unsigned int random_seed = 0;
      rng_t base_rng(random_seed);
      
      /*
      // (2**50 = 1T samples, 1000 chains)
      //static boost::uintmax_t DISCARD_STRIDE = static_cast<boost::uintmax_t>(1) << 50;
      
      // DISCARD_STRIDE <<= 50;
      //base_rng.discard(DISCARD_STRIDE * (chain_id - 1));
      */
      
      std::vector<double> q(model.num_params_r(), 1.0);
      std::vector<int> r;
      
      std::cout << model.log_prob(q, r) << std::endl;
      
      //stan::mcmc::unit_metric_hmc<Model, rng_t> sampler(model);
      //std::cout << "Before: " << q.at(0) << std::endl;
      //sampler.sample(q, r);
      //std::cout << "After: " << q.at(0) << std::endl;
      
      return 0;
      
    }

  } // namespace prob


} // namespace stan

#endif
