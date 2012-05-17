
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
namespace rstan {
  // wrap the args for running a model 
  // from Rcpp::List 
  class hmcargs {
  private:
    std::string samples; // the file for outputing the samples 
    unsigned int iter;   // number of iterations 
    unsigned int warmup; // number of warmup 
    unsigned int thin; 
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
    hmcargs(): 
      samples("samples.csv"),  
      iter(2000U), 
      warmup(1000U), 
      thin(1U), 
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
    hmcargs(Rcpp ::List) {
    } 
    int get_iter() {
      return iter; 
    } 
    std::string& get_samples() {
      return samples; 
    } 
    unsigned int get_warmup() {
      return warmup; 
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

