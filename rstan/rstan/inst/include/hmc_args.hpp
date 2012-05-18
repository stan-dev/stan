
#ifndef __RSTAN__HMCARGS_HPP__
#define __RSTAN__HMCARGS_HPP__


/* output from `anon_model --help' */
/*

  --data=<file>       Read data from specified dump-format file
                          (required if model declares data)

  --init=<file>       Use initial values from specified file or zero values if <file>=0
                          (default is random initialization)

  --samples=<file>    File into which samples are written
                          (default = samples.csv)

  --append_samples    Append samples to existing file if it exists
                          (does not write header in append mode)

  --seed=<int>        Random number generation seed
                          (default = randomly generated from time)

  --chain_id=<int>    Markov chain identifier
                          (default = 1)

  --iter=<+int>       Total number of iterations, including warmup
                          (default = 2000)

  --warmup=<+int>     Discard the specified number of initial samples
                          (default = iter / 2)

  --thin=<+int>       Period between saved samples after warm up
                          (default = max(1, floor(iter - warmup) / 1000))

  --refresh=<+int>    Period between samples updating progress report print
                          (default = max(1,iter/200)))

  --leapfrog_steps=<int>
                      Number of leapfrog steps; -1 for No-U-Turn adaptation
                          (default = -1)

  --max_treedepth=<int>
                      Limit NUTS leapfrog steps to 2^max_tree_depth; -1 for no limit
                          (default = 10)

  --epsilon=<float>   Initial value for step size, or -1 to set automatically
                          (default = -1)

  --epsilon_pm=<[0,1]>
                      Sample epsilon +/- epsilon * epsilon_pm
                          (default = 0.0)

  --epsilon_adapt_off
                      Turn off step size adaptation (default is on)

  --delta=<+float>    Initial step size for step-size adaptation
                          (default = 0.5)

  --gamma=<+float>    Gamma parameter for dual averaging step-size adaptation
                          (default = 0.05)

  --test_grad         Test gradient calculations using finite differences
*/


#include <Rcpp.h>
#include <rlist_util.hpp>
#include <Rinternals.h> 

namespace rstan {
  // wrap the arguments for hmc (including nuts) samplers 
  // from Rcpp::List 
  class hmc_args {
  private:
    std::string sample_file; // the file for outputing the samples 
    unsigned int iter;   // number of iterations 
    unsigned int warmup; // number of warmup 
    unsigned int thin; 
    unsigned int refresh; 
    int leapfrog_steps; 
    double epsilon; 
    int max_treedepth; 
    double epsilon_pm; 
    bool epsilon_adapt; 
    double delta; 
    double gamma; 
    int random_seed; 
    unsigned int chain_id; 
    bool append_samples; 
    bool test_grad; 
    std::string init; 
   
  public:
    hmc_args(): 
      sample_file("samples.csv"),  
      iter(2000U), 
      warmup(1000U), 
      thin(1U), 
      refresh(1U), 
      leapfrog_steps(-1), 
      epsilon(-1.0), 
      max_treedepth(10), 
      epsilon_pm(0.0), 
      epsilon_adapt(true), 
      delta(0.5), 
      gamma(0.05), 
      random_seed(std::time(0)), 
      chain_id(1), 
      append_samples(false), 
      test_grad(true), 
      init("random") {
    } 
    hmc_args(Rcpp::List &in) {
      /*
      std::vector<std::string> argnames 
        = Rcpp::as<std::vector<std::string> >(in.names()); 
      */
   
      SEXP tsexp = get_list_element_by_name(in, "sample_file"); 
      if (Rf_isNull(tsexp)) sample_file = "samples.csv"; 
      else sample_file = Rcpp::as<std::string>(tsexp); 

      tsexp = get_list_element_by_name(in, "iter"); 
      if (Rf_isNull(tsexp)) iter = 2000U;  
      else iter = Rcpp::as<unsigned int>(tsexp); 

      tsexp = get_list_element_by_name(in, "warmup"); 
      if (Rf_isNull(tsexp)) warmup = iter / 2; 
      else warmup = Rcpp::as<unsigned int>(tsexp); 

      tsexp = get_list_element_by_name(in, "thin"); 
      unsigned int calculated_thin = (iter - warmup) / 1000U;
      if (Rf_isNull(tsexp)) thin = (calculated_thin > 1) ? calculated_thin : 1U;
      else thin = Rcpp::as<unsigned int>(tsexp); 

      tsexp = get_list_element_by_name(in, "leapfrog_steps");
      if (Rf_isNull(tsexp)) leapfrog_steps = -1; 
      else leapfrog_steps = Rcpp::as<int>(tsexp); 

      tsexp = get_list_element_by_name(in, "epsilon"); 
      if (Rf_isNull(tsexp)) epsilon = -1.0; 
      else epsilon = Rcpp::as<double>(tsexp); 

      tsexp = get_list_element_by_name(in, "epsilon_pm"); 
      if (Rf_isNull(tsexp)) epsilon_pm = 0.0; 
      else epsilon_pm = Rcpp::as<double>(tsexp); 

      tsexp = get_list_element_by_name(in, "max_treedepth"); 
      if (Rf_isNull(tsexp))  max_treedepth = 10; 
      else max_treedepth = Rcpp::as<int>(tsexp); 
     
      tsexp = get_list_element_by_name(in, "epsilon_adapt"); 
      if (Rf_isNull(tsexp)) epsilon_adapt = true; 
      else epsilon_adapt = Rcpp::as<bool>(tsexp); 

      tsexp = get_list_element_by_name(in, "delta"); 
      if (Rf_isNull(tsexp))  delta = 0.5;
      else delta = Rcpp::as<double>(tsexp); 

      tsexp = get_list_element_by_name(in, "gamma"); 
      if (Rf_isNull(tsexp)) gamma = 0.05; 
      else gamma = Rcpp::as<double>(tsexp); 
      
      tsexp = get_list_element_by_name(in, "refresh"); 
      if (Rf_isNull(tsexp))  refresh = 1; 
      else refresh = Rcpp::as<unsigned int>(tsexp); 


      tsexp = get_list_element_by_name(in, "seed"); 
      if (Rf_isNull(tsexp)) random_seed = std::time(0); 
      else random_seed = Rcpp::as<unsigned int>(tsexp); 

      tsexp = get_list_element_by_name(in, "chain_id"); 
      if (Rf_isNull(tsexp)) chain_id = 1; 
      else chain_id = Rcpp::as<unsigned int>(tsexp); 

      
      tsexp = get_list_element_by_name(in, "init"); 
      if (Rf_isNull(tsexp)) init = "random"; 
      else init = Rcpp::as<std::string>(tsexp); // "0", "user", or "random"

      tsexp = get_list_element_by_name(in, "append_samples"); 
      if (Rf_isNull(tsexp)) append_samples = false; 
      else append_samples = Rcpp::as<bool>(tsexp); 

    } 
    int get_iter() {
      return iter; 
    } 
    std::string& get_sample_file() {
      return sample_file; 
    } 
    unsigned int get_warmup() {
      return warmup; 
    } 
    unsigned int get_refresh() { 
      return refresh; 
    } 
    unsigned int get_thin() {
      return thin;
    } 
    int get_leapfrog_steps() {
      return leapfrog_steps; 
    } 
    double get_epsilon() {
      return epsilon; 
    } 
    int get_max_treedepth() {
      return max_treedepth; 
    } 
    double get_epsilon_pm() {
      return epsilon; 
    } 
    bool get_epsilon_adapt() {
      return epsilon_adapt; 
    } 
    double get_delta() {  
      return delta;
    } 
    double get_gamma() { 
      return gamma;
    } 
    bool get_append_samples() {
      return append_samples; 
    } 
    bool get_test_grad() {
      return test_grad; 
    } 
    int get_random_seed() {
      return random_seed; 
    } 
   std::string get_init() {
      return init;
    } 
    unsigned int get_chain_id() {
      return chain_id; 
    } 
  }; 
} 

#endif 

