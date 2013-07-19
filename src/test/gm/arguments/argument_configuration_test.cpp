#include <stan/gm/arguments/argument_probe.hpp>
#include <stan/gm/arguments/arg_id.hpp>
#include <stan/gm/arguments/arg_data.hpp>
#include <stan/gm/arguments/arg_init.hpp>
#include <stan/gm/arguments/arg_random.hpp>
#include <stan/gm/arguments/arg_output.hpp>

#include <vector>

#include <gtest/gtest.h>


TEST(StanGmArgumentsConfiguration, Test) {
  
  std::vector<stan::gm::argument*> valid_arguments;
  valid_arguments.push_back(new stan::gm::arg_id());
  valid_arguments.push_back(new stan::gm::arg_data());
  valid_arguments.push_back(new stan::gm::arg_init());
  valid_arguments.push_back(new stan::gm::arg_random());
  valid_arguments.push_back(new stan::gm::arg_output());
  
  stan::gm::argument_probe probe(valid_arguments, "test/gm/arguments/");
  probe.probe_args();
  
  for (int i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
  EXPECT_EQ(1, 1);
  
}
