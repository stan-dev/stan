#include <stan/callbacks/dispatcher.hpp>
#include <stan/services/util/configure_dispatcher.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <memory>

class ConfigureDispatcherTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create our stringstreams as shared_ptrs
    sample_stream = std::make_shared<std::stringstream>();
    metric_stream = std::make_shared<std::stringstream>();
    diagnostic_stream = std::make_shared<std::stringstream>();
    init_stream = std::make_shared<std::stringstream>();
  }
  
  std::shared_ptr<std::stringstream> sample_stream;
  std::shared_ptr<std::stringstream> metric_stream;
  std::shared_ptr<std::stringstream> diagnostic_stream;
  std::shared_ptr<std::stringstream> init_stream;
};

TEST_F(ConfigureDispatcherTest, BasicFunctionality) {
  // Create a map of info_types to stream pointers
  std::unordered_map<stan::callbacks::info_type, std::shared_ptr<std::ostream>, 
                   stan::callbacks::info_type_hash> output_streams;
  
  // Add our streams to the map - cast to std::ostream base class
  output_streams[stan::callbacks::info_type::SAMPLE] = sample_stream;
  output_streams[stan::callbacks::info_type::METRIC] = metric_stream;
  output_streams[stan::callbacks::info_type::DIAGNOSTIC] = diagnostic_stream;
  output_streams[stan::callbacks::info_type::UNCONSTRAINED_INITS] = init_stream;
  
  // Call configure_dispatcher
  auto dispatcher = stan::services::util::configure_dispatcher(output_streams);
  
  // Test the functionality
  dispatcher.dispatch(stan::callbacks::info_type::SAMPLE, std::string("sample_message"));
  dispatcher.dispatch(stan::callbacks::info_type::DIAGNOSTIC, std::string("diagnostic_message"));
  
  dispatcher.begin_record(stan::callbacks::info_type::METRIC);
  dispatcher.dispatch(stan::callbacks::info_type::METRIC, "key", "value");
  dispatcher.end_record(stan::callbacks::info_type::METRIC);
  
  dispatcher.dispatch(stan::callbacks::info_type::UNCONSTRAINED_INITS, std::string("init_message"));
  
  // Verify output in our streams
  EXPECT_TRUE(sample_stream->str().find("sample_message") != std::string::npos);
  EXPECT_TRUE(diagnostic_stream->str().find("diagnostic_message") != std::string::npos);
  EXPECT_TRUE(metric_stream->str().find("key") != std::string::npos);
  EXPECT_TRUE(metric_stream->str().find("value") != std::string::npos);
  EXPECT_TRUE(init_stream->str().find("init_message") != std::string::npos);
}

TEST_F(ConfigureDispatcherTest, NullStream) {
  std::unordered_map<stan::callbacks::info_type, std::shared_ptr<std::ostream>, 
                   stan::callbacks::info_type_hash> output_streams;
  
  // Add a null shared_ptr
  output_streams[stan::callbacks::info_type::SAMPLE] = nullptr;
  
  // Should throw an exception
  EXPECT_THROW(
    stan::services::util::configure_dispatcher(output_streams),
    std::runtime_error
  );
}

TEST_F(ConfigureDispatcherTest, MoveSemantics) {
  // Create a map of info_types to stream pointers
  std::unordered_map<stan::callbacks::info_type, std::shared_ptr<std::ostream>, 
                   stan::callbacks::info_type_hash> output_streams;
  
  // Add our streams to the map
  output_streams[stan::callbacks::info_type::SAMPLE] = sample_stream;
  
  // Create first dispatcher
  auto dispatcher1 = stan::services::util::configure_dispatcher(output_streams);
  
  // Move dispatcher to a new object
  auto dispatcher2 = std::move(dispatcher1);
  
  // Test that the moved dispatcher works
  dispatcher2.dispatch(stan::callbacks::info_type::SAMPLE, std::string("moved_message"));
  
  // Verify output
  EXPECT_TRUE(sample_stream->str().find("moved_message") != std::string::npos);
}

// Test that resources are properly managed
TEST_F(ConfigureDispatcherTest, ResourceManagement) {
  // Create a scope for the dispatcher
  {
    std::unordered_map<stan::callbacks::info_type, std::shared_ptr<std::ostream>, 
                     stan::callbacks::info_type_hash> output_streams;
    
    // Add our streams to the map
    output_streams[stan::callbacks::info_type::SAMPLE] = sample_stream;
    
    // Create the dispatcher (will go out of scope at the end of this block)
    auto dispatcher = stan::services::util::configure_dispatcher(output_streams);
    
    // Use the dispatcher
    dispatcher.dispatch(stan::callbacks::info_type::SAMPLE, std::string("test_message"));
  }
  
  // The stream should still be valid and contain our message
  EXPECT_TRUE(sample_stream->good());
  EXPECT_TRUE(sample_stream->str().find("test_message") != std::string::npos);
}
