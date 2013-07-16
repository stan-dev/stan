#include <gtest/gtest.h>
#include <test/models/utility.hpp>

TEST(gm,compile_models) {
  SUCCEED() 
    << "Model compilation done through makefile dependencies." << std::endl
    << "Should have compiled: src/test/gm/model_specs/compiled/*.stan";
}

TEST(gm,issue91_segfault_printing_uninitialized) {
  char path_separator = get_path_separator();
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("issue91");

  std::string command 
    = convert_model_path(model_path)
    + " --iter=0" 
    + " --samples=" + convert_model_path(model_path) + ".csv";
  
  run_command(command);

  SUCCEED()
    << "running this model should not seg fault";
}
