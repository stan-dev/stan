 #ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__

#include <fstream>
#include <stdexcept>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/mcmc_writer.hpp>

#include <stan/gm/arguments/argument_parser.hpp>
#include <stan/gm/arguments/arg_id.hpp>
#include <stan/gm/arguments/arg_data.hpp>
#include <stan/gm/arguments/arg_init.hpp>
#include <stan/gm/arguments/arg_random.hpp>
#include <stan/gm/arguments/arg_output.hpp>

#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include <stan/model/util.hpp>

#include <stan/optimization/newton.hpp>
#include <stan/optimization/nesterov_gradient.hpp>
#include <stan/optimization/bfgs.hpp>

namespace stan {

  namespace gm {

    void write_stan(std::ostream* s, const std::string prefix = "") {
      if (!s) return;
      
      *s << prefix << " stan_version_major = " << stan::MAJOR_VERSION << std::endl;
      *s << prefix << " stan_version_minor = " << stan::MINOR_VERSION << std::endl;
      *s << prefix << " stan_version_patch = " << stan::PATCH_VERSION << std::endl;
      
    }
    
    void write_model(std::ostream* s, const std::string model_name, const std::string prefix = "") {
      if (!s) return;
      
      *s << prefix << " model = " << model_name << std::endl;
      
    }
    
    void write_error_msg(std::ostream* error_stream,
                         const std::exception& e) {
      
      if (!error_stream) return;
      
      *error_stream << std::endl
                    << "Informational Message: The current Metropolis proposal is about to be"
                    << " rejected becuase of the following issue:"
                    << std::endl
                    << e.what() << std::endl
                    << "If this warning occurs sporadically, such as for highly constrained"
                    << " variable types like covariance matrices, then the sampler is fine,"
                    << std::endl
                    << "but if this warning occurs often then your model may be either"
                    << " severely ill-conditioned or misspecified."
                    << std::endl;
      
    }
    
    /** 
     * Indicates whether it should print on the current iteration.
     * The function returns:
     *   true if refresh > 0 and 
     *     (n == 0 or (n + 1) % refresh == 0 or special == true)
     *   false otherwise.
     * Examples:
     *   if refresh = 0, always returns false.
     *   if refresh = 10 and special == false, 
     *      returns true for n = {0, 9, 19, ...},
     *      returns false for n = {1 - 8, 10 - 18, 20 - 28, ...}.
     *   if refresh = 10 and special == true, returns true for all n.
     *
     * @param n Iteration number
     * @param special When true, returns true
     * @param refresh Number of iterations to refresh
     */
    bool do_print(int n, bool special, int refresh) {
      return (refresh > 0) &&
        (special || n == 0 || ((n + 1) % refresh == 0) );
    }
    
    /**
     * Indicates whether it should print on the current iteration.
     * The function returns:
     *   true if refresh > 0 and (n == 0 or (n + 1) % refresh == 0)
     *   false otherwise.
     * Examples:
     *   if refresh = 0, always returns false.
     *   if refresh = 10, returns true for n = {0, 9, 19, ...},
     *      returns false for n = {1 - 8, 10 - 18, 20 - 28, ...}.
     * 
     * @param n Iteration number
     * @param refresh Number of iterations to refresh
     */
    bool do_print(int n, int refresh) {
      return do_print(n,false,refresh);
    }

    void print_progress(int m, int start, int finish, int refresh, bool warmup) {
      
      int it_print_width = std::ceil(std::log10(finish));

      if (do_print(m, (start + m + 1 == finish), refresh )) {
        
        std::cout << "Iteration: ";
        std::cout << std::setw(it_print_width) << m + 1 + start
                  << " / " << finish;
          
        std::cout << " [" << std::setw(3) 
                  << static_cast<int>( (100.0 * (start + m + 1)) / finish )
                  << "%] ";
        std::cout << (warmup ? " (Warmup)" : " (Sampling)");
        std::cout << std::endl;
      }
    
    }
    
    template <class Model, class RNG>
    void run_markov_chain(stan::mcmc::base_mcmc* sampler,
                          int num_iterations,
                          int start,
                          int finish,
                          int num_thin,
                          int refresh,
                          bool save,
                          bool warmup,
                          stan::io::mcmc_writer<Model>& writer,
                          stan::mcmc::sample& init_s,
                          Model& model,
                          RNG& base_rng) {
      
      for (int m = 0; m < num_iterations; ++m) {
      
        print_progress(m, start, finish, refresh, warmup);
      
        init_s = sampler->transition(init_s);
          
        if ( save && ( (m % num_thin) == 0) ) {
          writer.print_sample_params(base_rng, init_s, *sampler, model);
          writer.print_diagnostic_params(init_s, sampler);
        }

      }
      
    }

    template <class Model, class RNG>
    void warmup(stan::mcmc::base_mcmc* sampler,
                int num_warmup,
                int num_samples,
                int num_thin,
                int refresh,
                bool save,
                stan::io::mcmc_writer<Model>& writer,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Model, RNG>(sampler, num_warmup, 0, num_warmup + num_samples, num_thin,
                                   refresh, save, true,
                                   writer,
                                   init_s, model, base_rng);
      
    }

    template <class Model, class RNG>
    void sample(stan::mcmc::base_mcmc* sampler,
                int num_warmup,
                int num_samples,
                int num_thin,
                int refresh,
                bool save,
                stan::io::mcmc_writer<Model>& writer,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Model, RNG>(sampler, num_samples, num_warmup, 
                                   num_warmup + num_samples, num_thin,
                                   refresh, save, false,
                                   writer,
                                   init_s, model, base_rng);
      
    }
      
    
    template<class Sampler>
    bool init_static_hmc(stan::mcmc::base_mcmc* sampler, argument* algorithm) {

      categorical_argument* hmc = dynamic_cast<categorical_argument*>(
                                  algorithm->arg("hmc"));
      
      categorical_argument* base = dynamic_cast<categorical_argument*>(
                                   algorithm->arg("hmc")->arg("engine")->arg("static"));
      
      double epsilon = dynamic_cast<real_argument*>(hmc->arg("stepsize"))->value();
      double epsilon_jitter 
        = dynamic_cast<real_argument*>(hmc->arg("stepsize_jitter"))->value();
      double int_time = dynamic_cast<real_argument*>(base->arg("int_time"))->value();
      
      dynamic_cast<Sampler*>(sampler)->set_nominal_stepsize_and_T(epsilon, int_time);
      dynamic_cast<Sampler*>(sampler)->set_stepsize_jitter(epsilon_jitter);
      
      
      return true;
      
    }
  
    template<class Sampler>
    bool init_nuts(stan::mcmc::base_mcmc* sampler, argument* algorithm) {
      
      categorical_argument* hmc = dynamic_cast<categorical_argument*>(
                                   algorithm->arg("hmc"));
      
      categorical_argument* base = dynamic_cast<categorical_argument*>(
                                   algorithm->arg("hmc")->arg("engine")->arg("nuts"));

      double epsilon = dynamic_cast<real_argument*>(hmc->arg("stepsize"))->value();
      double epsilon_jitter 
        = dynamic_cast<real_argument*>(hmc->arg("stepsize_jitter"))->value();
      int max_depth = dynamic_cast<int_argument*>(base->arg("max_depth"))->value();
      
      dynamic_cast<Sampler*>(sampler)->set_nominal_stepsize(epsilon);
      dynamic_cast<Sampler*>(sampler)->set_stepsize_jitter(epsilon_jitter);
      dynamic_cast<Sampler*>(sampler)->set_max_depth(max_depth);
      
      return true;
      
    }

    template<class Sampler>
    bool init_adapt(Sampler* sampler, 
                    const double delta,
                    const double gamma,
                    const double kappa, 
                    const double t0,
                    const Eigen::VectorXd& cont_params) {
      const double epsilon = sampler->get_nominal_stepsize();
      
      sampler->get_stepsize_adaptation().set_mu(log(10 * epsilon));
      sampler->get_stepsize_adaptation().set_delta(delta);
      sampler->get_stepsize_adaptation().set_gamma(gamma);
      sampler->get_stepsize_adaptation().set_kappa(kappa);
      sampler->get_stepsize_adaptation().set_t0(t0);
      
      sampler->engage_adaptation();
      
      try {
        sampler->z().q = cont_params;
        sampler->init_stepsize();
      } catch (const std::exception& e) {
        std::cout << "Exception initializing step size." << std::endl
                  << e.what() << std::endl;
        return false;
      }
      
      return true;
    }

    template<class Sampler>
    bool init_adapt(stan::mcmc::base_mcmc* sampler, categorical_argument* adapt,
                    const Eigen::VectorXd& cont_params) {
      
      double delta = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
      double gamma = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
      double kappa = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
      double t0    = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();
      
      Sampler* s = dynamic_cast<Sampler*>(sampler);

      return init_adapt<Sampler>(s, delta, gamma, kappa, t0, cont_params);
    }
    
    template<class Sampler>
    bool init_windowed_adapt(stan::mcmc::base_mcmc* sampler, categorical_argument* adapt, 
                             unsigned int num_warmup, const Eigen::VectorXd& cont_params) {
      
      init_adapt<Sampler>(sampler, adapt, cont_params);
      
      unsigned int init_buffer 
        = dynamic_cast<u_int_argument*>(adapt->arg("init_buffer"))->value();
      unsigned int term_buffer 
        = dynamic_cast<u_int_argument*>(adapt->arg("term_buffer"))->value();
      unsigned int window = dynamic_cast<u_int_argument*>(adapt->arg("window"))->value();
      
      dynamic_cast<Sampler*>(sampler)->set_window_params(num_warmup, init_buffer, 
                                                         term_buffer, window, &std::cout);
      
      return true;
      
    }
    
    template <class Model>
    int command(int argc, const char* argv[]) {

      std::vector<argument*> valid_arguments;
      valid_arguments.push_back(new arg_id());
      valid_arguments.push_back(new arg_data());
      valid_arguments.push_back(new arg_init());
      valid_arguments.push_back(new arg_random());
      valid_arguments.push_back(new arg_output());
      
      argument_parser parser(valid_arguments);
      int err_code = parser.parse_args(argc, argv, &std::cout, &std::cout);

      if (err_code != 0) {
        std::cout << "Failed to parse arguments, terminating Stan" << std::endl;
        return err_code;
      }
      
      if (parser.help_printed())
        return err_code;

      parser.print(&std::cout);
      std::cout << std::endl;
      
      // Identification
      unsigned int id = dynamic_cast<int_argument*>(parser.arg("id"))->value();
      
      //////////////////////////////////////////////////
      //            Random number generator           //
      //////////////////////////////////////////////////
      
      unsigned int random_seed = 0;
      u_int_argument* random_arg 
        = dynamic_cast<u_int_argument*>(parser.arg("random")->arg("seed"));
      
      if (random_arg->is_default()) {
        random_seed = (boost::posix_time::microsec_clock::universal_time() -
                       boost::posix_time::ptime(boost::posix_time::min_date_time))
                      .total_milliseconds();
        
        random_arg->set_value(random_seed);
        
      } else {
        random_seed = random_arg->value();
      }
      
      typedef boost::ecuyer1988 rng_t; // (2**50 = 1T samples, 1000 chains)
      rng_t base_rng(random_seed);
      
      // Advance generator to avoid process conflicts
      static boost::uintmax_t DISCARD_STRIDE = static_cast<boost::uintmax_t>(1) << 50;
      base_rng.discard(DISCARD_STRIDE * (id - 1));
      
      //////////////////////////////////////////////////
      //                  Input/Output                //
      //////////////////////////////////////////////////
      
      // Data input
      std::string data_file 
        = dynamic_cast<string_argument*>(parser.arg("data")->arg("file"))->value();
      
      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();
      
      // Sample output
      std::string output_file = dynamic_cast<string_argument*>(
                                parser.arg("output")->arg("file"))->value();
      std::fstream* output_stream = 0;
      if (output_file != "") {
        output_stream = new std::fstream(output_file.c_str(),
                                         std::fstream::out);
      }
      
      // Diagnostic output
      std::string diagnostic_file = dynamic_cast<string_argument*>(
                                    parser.arg("output")->arg("diagnostic_file"))->value();
      
      std::fstream* diagnostic_stream = 0;
      if (diagnostic_file != "") {
        diagnostic_stream = new std::fstream(diagnostic_file.c_str(),
                                             std::fstream::out);
      }
      
      // Refresh rate
      int refresh = dynamic_cast<int_argument*>(
                    parser.arg("output")->arg("refresh"))->value();
      
      //////////////////////////////////////////////////
      //                Initialize Model              //
      //////////////////////////////////////////////////
      
      Model model(data_var_context, &std::cout);
      
      Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(model.num_params_r());

      if (output_stream) {
        write_stan(output_stream, "#");
        write_model(output_stream, model.model_name(), "#");
        parser.print(output_stream, "#");
      }
      
      if (diagnostic_stream) {
        write_stan(diagnostic_stream, "#");
        write_model(diagnostic_stream, model.model_name(), "#");
        parser.print(diagnostic_stream, "#");
      }
      
      int num_init_tries = -1;
      
      std::string init = dynamic_cast<string_argument*>(
                         parser.arg("init"))->value();
      
      try {
        
        double R = std::fabs(boost::lexical_cast<double>(init));
        
        if (R == 0) {
          
          cont_params.setZero();
          
          double init_log_prob;
          Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());
          
          try {
            init_log_prob               
              = stan::model::log_prob_grad<true, true>(model, cont_params, init_grad, &std::cout);
            //stan::model::gradient(model, cont_params, init_log_prob, init_grad, &std::cout);
          } catch (const std::exception& e) {
            std::cout << "Rejecting initialization at zero because of gradient failure."
                      << std::endl 
                      << e.what() << std::endl;
            return error_codes::OK;
          }
          
          if (!boost::math::isfinite(init_log_prob)) {
            std::cout << "Rejecting initialization at zero because of vanishing density." 
                      << std::endl;
            return 0;
          }
          
          for (int i = 0; i < init_grad.size(); ++i) {
            if (!boost::math::isfinite(init_grad[i])) {
              std::cout << "Rejecting initialization at zero because of divergent gradient."
                        << std::endl;
              return 0;
            }
          }

        } else {
          
          boost::random::uniform_real_distribution<double>
          init_range_distribution(-R, R);
          
          boost::variate_generator<rng_t&,
          boost::random::uniform_real_distribution<double> >
          init_rng(base_rng, init_range_distribution);
          
          cont_params.setZero();
          
          // Random initializations until log_prob is finite
          Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());
          static int MAX_INIT_TRIES = 100;
          
          for (num_init_tries = 1; num_init_tries <= MAX_INIT_TRIES; ++num_init_tries) {
            
            for (int i = 0; i < cont_params.size(); ++i)
              cont_params(i) = init_rng();
            
            // FIXME: allow config vs. std::cout
            double init_log_prob;
            try {
              init_log_prob
                = stan::model::log_prob_grad<true, true>(model, cont_params, init_grad, &std::cout);
              //stan::model::gradient(model, cont_params, init_log_prob, init_grad, &std::cout);
            } catch (const std::exception& e) {
              write_error_msg(&std::cout, e);
              std::cout << "Rejecting proposed initial value with zero density." 
                        << std::endl;
              init_log_prob = -std::numeric_limits<double>::infinity();
            }
            
            if (!boost::math::isfinite(init_log_prob))
              continue;
            for (int i = 0; i < init_grad.size(); ++i)
              if (!boost::math::isfinite(init_grad(i)))
                continue;
            break;
            
          }
          
          if (num_init_tries > MAX_INIT_TRIES) {
            std::cout << std::endl << std::endl
                      << "Initialization between (" << -R << ", " << R << ") failed after "
                      << MAX_INIT_TRIES << " attempts. " << std::endl;
            std::cout << " Try specifying initial values,"
                      << " reducing ranges of constrained values,"
                      << " or reparameterizing the model."
                      << std::endl;
            return -1;
          }
          
        }
        
      } catch(...) {
      
        try {
        
          std::fstream init_stream(init.c_str(), std::fstream::in);
          if (init_stream.fail()) {
            std::string msg("ERROR: specified initialization file does not exist: ");
            msg += init;
            throw std::invalid_argument(msg);
          }
          
          stan::io::dump init_var_context(init_stream);
          init_stream.close();
          model.transform_inits(init_var_context, cont_params);
        
        } catch (const std::exception& e) {
          std::cerr << "Error during user-specified initialization:" << std::endl
                    << e.what() << std::endl;
          return -5;
        }
        
        double init_log_prob;
        Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());
        
        try {
          init_log_prob
            = stan::model::log_prob_grad<true, true>(model, cont_params, init_grad, &std::cout);
          stan::model::gradient(model, cont_params, init_log_prob, init_grad, &std::cout);
        } catch (const std::exception& e) {
          std::cout 
            << "Rejecting user-specified initialization because of gradient failure."
            << std::endl
            << e.what() << std::endl;
          return 0;
        }
        
        if (!boost::math::isfinite(init_log_prob)) {
          std::cout 
            << "Rejecting user-specified initialization because of vanishing density."
            << std::endl;
          return 0;
        }
        
        for (int i = 0; i < init_grad.size(); ++i) {
          if (!boost::math::isfinite(init_grad[i])) {
            std::cout 
              << "Rejecting user-specified initialization because of divergent gradient."
              << std::endl;
            return 0;
          }
        }
        
      }
      
      //////////////////////////////////////////////////
      //               Model Diagnostics              //
      //////////////////////////////////////////////////
      
      if (parser.arg("method")->arg("diagnose")) {
      
        std::vector<double> cont_vector(cont_params.size());
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector.at(i) = cont_params(i);
        std::vector<int> disc_vector;
        
        list_argument* test = dynamic_cast<list_argument*>
                              (parser.arg("method")->arg("diagnose")->arg("test"));
        
        if (test->value() == "gradient") {
          std::cout << std::endl << "TEST GRADIENT MODE" << std::endl;

          double epsilon = dynamic_cast<real_argument*>
                           (test->arg("gradient")->arg("epsilon"))->value();
          
          double error = dynamic_cast<real_argument*>
                         (test->arg("gradient")->arg("error"))->value();
          
          int num_failed
            = stan::model::test_gradients<true,true>(model,cont_vector, disc_vector, 
                                                     epsilon, error, std::cout);
          
          if (output_stream) {
            num_failed
              = stan::model::test_gradients<true,true>(model,cont_vector, disc_vector,
                                                       epsilon, error, *output_stream);
          }
          
          if (diagnostic_stream) {
            num_failed
              = stan::model::test_gradients<true,true>(model,cont_vector, disc_vector, 
                                                       epsilon, error, *diagnostic_stream);
          }
          
          (void) num_failed; // FIXME: do something with the number failed
          
          return error_codes::OK;
        }
        
      }
      
      //////////////////////////////////////////////////
      //           Optimization Algorithms            //
      //////////////////////////////////////////////////
      
      if (parser.arg("method")->arg("optimize")) {
        
        std::vector<double> cont_vector(cont_params.size());
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector.at(i) = cont_params(i);
        std::vector<int> disc_vector;
        
        list_argument* algo = dynamic_cast<list_argument*>
                              (parser.arg("method")->arg("optimize")->arg("algorithm"));

        int num_iterations = dynamic_cast<int_argument*>(
                             parser.arg("method")->arg("optimize")->arg("iter"))->value();

        bool save_iterations 
          = dynamic_cast<bool_argument*>(parser.arg("method")
                                         ->arg("optimize")
                                         ->arg("save_iterations"))->value();
        if (output_stream) {
          *output_stream << "lp__,";
          model.write_csv_header(*output_stream);
        }

        double lp(0);
        int return_code = error_codes::CONFIG;
        if (algo->value() == "nesterov") {
          bool epsilon = dynamic_cast<real_argument*>(
                         algo->arg("nesterov")->arg("stepsize"))->value();
          
          
          stan::optimization::NesterovGradient<Model> ng(model, cont_vector, disc_vector,
                                                         epsilon, &std::cout);
          
          lp = ng.logp();
          
          double lastlp = lp - 1;
          std::cout << "Initial log joint probability = " << lp << std::endl;
          if (output_stream && save_iterations) {
            *output_stream << lp << ',';
            model.write_csv(base_rng, cont_vector, disc_vector,
                            *output_stream, &std::cout);
            output_stream->flush();
          }

          int m = 0;
          for (int i = 0; i < num_iterations; i++) {
            lastlp = lp;
            lp = ng.step();
            ng.params_r(cont_vector);
            if (do_print(i, refresh)) {
              std::cout << "Iteration ";
              std::cout << std::setw(2) << (m + 1) << ". ";
              std::cout << "Log joint probability = " << std::setw(10) << lp;
              std::cout << ". Improved by " << (lp - lastlp) << ".";
              std::cout << std::endl;
              std::cout.flush();
            }
            m++;
            if (output_stream && save_iterations) {
              *output_stream << lp << ',';
              model.write_csv(base_rng, cont_vector, disc_vector,
                              *output_stream, &std::cout);
              output_stream->flush();
            }

          }
          return_code = error_codes::OK;
        } else if (algo->value() == "newton") {
          
          std::vector<double> gradient;
          try {
            lp = model.template log_prob<false, false>(cont_vector, disc_vector, &std::cout);
          } catch (const std::exception& e) {
            write_error_msg(&std::cout, e);
            lp = -std::numeric_limits<double>::infinity();
          }
          
          std::cout << "initial log joint probability = " << lp << std::endl;
          if (output_stream && save_iterations) {
            *output_stream << lp << ',';
            model.write_csv(base_rng, cont_vector, disc_vector,
                            *output_stream, &std::cout);
            output_stream->flush();
          }

          double lastlp = lp - 1;
          int m = 0;
          while ((lp - lastlp) / fabs(lp) > 1e-8) {
            
            lastlp = lp;
            lp = stan::optimization::newton_step(model, cont_vector, disc_vector);
            std::cout << "Iteration ";
            std::cout << std::setw(2) << (m + 1) << ". ";
            std::cout << "Log joint probability = " << std::setw(10) << lp;
            std::cout << ". Improved by " << (lp - lastlp) << ".";
            std::cout << std::endl;
            std::cout.flush();
            m++;

            if (output_stream && save_iterations) {
              *output_stream << lp << ',';
              model.write_csv(base_rng, cont_vector, disc_vector,
                              *output_stream, &std::cout);
            }
            
          }
          return_code = error_codes::OK;
        } else if (algo->value() == "bfgs") {
          
          stan::optimization::BFGSLineSearch<Model> bfgs(model, cont_vector, disc_vector,
                                                         &std::cout);
          bfgs._opts.alpha0 = dynamic_cast<real_argument*>(
                         algo->arg("bfgs")->arg("init_alpha"))->value();
          bfgs._opts.tolF = dynamic_cast<real_argument*>(
                         algo->arg("bfgs")->arg("tol_obj"))->value();
          bfgs._opts.tolGrad = dynamic_cast<real_argument*>(
                         algo->arg("bfgs")->arg("tol_grad"))->value();
          bfgs._opts.tolX = dynamic_cast<real_argument*>(
                         algo->arg("bfgs")->arg("tol_param"))->value();
          bfgs._opts.maxIts = num_iterations;
          
          lp = bfgs.logp();
          
          std::cout << "initial log joint probability = " << lp << std::endl;
          if (output_stream && save_iterations) {
            *output_stream << lp << ',';
            model.write_csv(base_rng, cont_vector, disc_vector,
                            *output_stream, &std::cout);
            output_stream->flush();
          }

          int ret = 0;
          
          while (ret == 0) {  
            if (do_print(bfgs.iter_num(), 50*refresh)) {
              std::cout << "    Iter ";
              std::cout << "     log prob ";
              std::cout << "       ||dx|| ";
              std::cout << "     ||grad|| ";
              std::cout << "      alpha ";
// MAB: commented out but left in because it may be useful for debugging in the future
//              std::cout << "     alpha0 ";
              std::cout << " # evals ";
              std::cout << " Notes " << std::endl;
            }
            
            ret = bfgs.step();
            lp = bfgs.logp();
            bfgs.params_r(cont_vector);
            
            if (do_print(bfgs.iter_num(), ret != 0 || !bfgs.note().empty(),refresh)) {
              std::cout << " " << std::setw(7) << bfgs.iter_num() << " ";
              std::cout << " " << std::setw(12) << std::setprecision(6) 
                        << lp << " ";
              std::cout << " " << std::setw(12) << std::setprecision(6) 
                        << bfgs.prev_step_size() << " ";
              std::cout << " " << std::setw(12) << std::setprecision(6) 
                        << bfgs.curr_g().norm() << " ";
              std::cout << " " << std::setw(10) << std::setprecision(4) 
                        << bfgs.alpha() << " ";
              // std::cout << " " << std::setw(10) << std::setprecision(4) 
              // << bfgs.alpha0() << " ";
              std::cout << " " << std::setw(7) 
                        << bfgs.grad_evals() << " ";
              std::cout << " " << bfgs.note() << " ";
              std::cout << std::endl;
            }
            
            if (output_stream && save_iterations) {
              *output_stream << lp << ',';
              model.write_csv(base_rng, cont_vector, disc_vector,
                              *output_stream, &std::cout);
              output_stream->flush();
            }
          }
          
          
          if (ret >= 0) {
            std::cout << "Optimization terminated normally: " << std::endl;
            return_code = error_codes::OK;
          } else {
            std::cout << "Optimization terminated with error: " << std::endl;
            return_code = error_codes::SOFTWARE;
          }
          std::cout << "  " << bfgs.get_code_string(ret) << std::endl;
        } else {
          return_code = error_codes::CONFIG;
        }

        if (output_stream) {
          *output_stream << lp << ',';
          model.write_csv(base_rng, cont_vector, disc_vector,
                          *output_stream, &std::cout);
          output_stream->flush();
          output_stream->close();
          delete output_stream;
        }
        return return_code;
      }
        
      //////////////////////////////////////////////////
      //              Sampling Algorithms             //
      //////////////////////////////////////////////////
      
      if (parser.arg("method")->arg("sample")) {
        
        // Check timing
        clock_t start_check = clock();
        
        double init_log_prob;
        Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());
        stan::model::log_prob_grad<true, true>(model, cont_params, init_grad, &std::cout);
        
        //stan::model::gradient(model, cont_params, init_log_prob, init_grad, &std::cout);
        
        clock_t end_check = clock();
        double deltaT = (double)(end_check - start_check) / CLOCKS_PER_SEC;
        
        std::cout << std::endl;
        std::cout << "Gradient evaluation took " << deltaT << " seconds" << std::endl;
        std::cout << "1000 transitions using 10 leapfrog steps per transition would take "
                  << 1e4 * deltaT << " seconds." << std::endl;
        std::cout << "Adjust your expectations accordingly!" << std::endl << std::endl;
        std::cout << std::endl;
        
        stan::io::mcmc_writer<Model> writer(output_stream, diagnostic_stream, &std::cout);
        
        // Sampling parameters
        int num_warmup = dynamic_cast<int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_warmup"))->value();
        
        int num_samples = dynamic_cast<int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_samples"))->value();
        
        int num_thin = dynamic_cast<int_argument*>(
                       parser.arg("method")->arg("sample")->arg("thin"))->value();
        
        bool save_warmup = dynamic_cast<bool_argument*>(
                           parser.arg("method")->arg("sample")->arg("save_warmup"))->value();
        
        stan::mcmc::sample s(cont_params, 0, 0);
        
        double warmDeltaT;
        double sampleDeltaT;
        
        // Sampler
        stan::mcmc::base_mcmc* sampler_ptr = 0;
        
        list_argument* algo = dynamic_cast<list_argument*>
                              (parser.arg("method")->arg("sample")->arg("algorithm"));
        
        categorical_argument* adapt = dynamic_cast<categorical_argument*>(
                                      parser.arg("method")->arg("sample")->arg("adapt"));
        bool adapt_engaged = dynamic_cast<bool_argument*>(adapt->arg("engaged"))->value();
        
        if (algo->value() == "fixed_param") {
          
          sampler_ptr = new stan::mcmc::fixed_param_sampler();
          
          adapt_engaged = false;
          
          if (num_warmup != 0) {
            std::cout << "Warning: warmup will be skipped for the fixed parameter sampler!" << std::endl;
            num_warmup = 0;
          }
          
        } else if (algo->value() == "rwm") {
          
          std::cout << algo->arg("rwm")->description() << std::endl;
          return 0;
        
        } else if (algo->value() == "hmc") {
          
          int engine_index = 0;
          list_argument* engine 
            = dynamic_cast<list_argument*>(algo->arg("hmc")->arg("engine"));
          if (engine->value() == "static") {
            engine_index = 0;
          } else if (engine->value() == "nuts") {
            engine_index = 1;
          }
          
          int metric_index = 0;
          list_argument* metric 
            = dynamic_cast<list_argument*>(algo->arg("hmc")->arg("metric"));
          if (metric->value() == "unit_e") {
            metric_index = 0;
          } else if (metric->value() == "diag_e") {
            metric_index = 1;
          } else if (metric->value() == "dense_e") {
            metric_index = 2;
          }
          
          int sampler_select = engine_index 
            + 10 * metric_index 
            + 100 * static_cast<int>(adapt_engaged);
          
          switch (sampler_select) {
              
            case 0: {
              typedef stan::mcmc::unit_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
              
            case 1: {        
              typedef stan::mcmc::unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
              
            case 10: {
              typedef stan::mcmc::diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 11: {
              typedef stan::mcmc::diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 20: {
              typedef stan::mcmc::dense_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 21: {
              typedef stan::mcmc::dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 100: {
              typedef stan::mcmc::adapt_unit_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt, cont_params)) return 0;
              break;
            }
            
            case 101: {
              typedef stan::mcmc::adapt_unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt, cont_params)) return 0;
              break;
            }
            
            case 110: {
              typedef stan::mcmc::adapt_diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              if (!init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params)) 
                return 0;
              break;
            }
            
            case 111: {
              typedef stan::mcmc::adapt_diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              if (!init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params)) 
                return 0;
              break;
            }
            
            case 120: {
              typedef stan::mcmc::adapt_dense_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              if (!init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params)) 
                return 0;
              break;
            }
            
            case 121: {
              typedef stan::mcmc::adapt_dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              if (!init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params))
                return 0;
              break;
            }
            
            default:
              std::cout << "No sampler matching HMC specification!" << std::endl;
              return 0;
          }
          
        }
        
        // Headers
        writer.print_sample_names(s, sampler_ptr, model);
        writer.print_diagnostic_names(s, sampler_ptr, model);
        
        // Warm-Up
        clock_t start = clock();
        
        warmup<Model, rng_t>(sampler_ptr, num_warmup, num_samples, num_thin,
                             refresh, save_warmup,
                             writer,
                             s, model, base_rng);
        
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        if (adapt_engaged) {
          dynamic_cast<mcmc::base_adapter*>(sampler_ptr)->disengage_adaptation();
          writer.print_adapt_finish(sampler_ptr);
        }
        
        // Sampling
        start = clock();
        
        stan::gm::sample<Model, rng_t>(sampler_ptr, num_warmup, num_samples, num_thin,
                                       refresh, true,
                                       writer,
                                       s, model, base_rng);
        
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        writer.print_timing(warmDeltaT, sampleDeltaT);
        
        if (sampler_ptr) delete sampler_ptr;
        
      }
      
      if (output_stream) {
        output_stream->close();
        delete output_stream;
      }
        
      if (diagnostic_stream) {
        diagnostic_stream->close();
        delete diagnostic_stream;
      }
      
      for (size_t i = 0; i < valid_arguments.size(); ++i)
        delete valid_arguments.at(i);
      
      return 0;
 
    }

  } // namespace gm

} // namespace stan

#endif
