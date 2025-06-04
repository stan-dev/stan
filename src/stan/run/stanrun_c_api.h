#ifndef STANRUN_C_API_H
#define STANRUN_C_API_H

#include <stddef.h>

/* Export macro for shared library visibility */
#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef STANRUN_BUILDING_LIBRARY
    #define STANRUN_API __declspec(dllexport)
  #else
    #define STANRUN_API __declspec(dllimport)
  #endif
#else
  #if defined(STANRUN_BUILDING_LIBRARY) && defined(__GNUC__)
    #define STANRUN_API __attribute__((visibility("default")))
  #else
    #define STANRUN_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define STANRUN_SUCCESS 0
#define STANRUN_ERROR_PARSING 1
#define STANRUN_ERROR_MODEL_LOAD 2
#define STANRUN_ERROR_SAMPLING 3
#define STANRUN_ERROR_INVALID_ARGS 4
#define STANRUN_ERROR_RUNTIME 5

/* Load a Stan model with optional data file
 * 
 * @param data_filename Path to data file, or empty string if no data needed
 * @param seed Random seed for model initialization
 * @param error_message Buffer to store error message on failure
 * @param error_message_size Size of error message buffer
 * @return Opaque model handle on success, NULL on failure
 */
STANRUN_API void* stosh_load_model(const char* data_filename, 
                                   unsigned int seed,
                                   char* error_message, 
                                   size_t error_message_size);

/* Run samplers on a loaded model using key-value parameter pairs
 * 
 * @param handle_ptr Handle returned from stosh_load_model
 * @param keys Array of parameter name strings
 * @param values Array of parameter value strings (parallel to keys)
 * @param num_params Number of key-value pairs
 * @param output_dir Buffer to receive actual output directory path used
 * @param output_dir_size Size of output directory buffer
 * @param error_message Buffer to store error message on failure
 * @param error_message_size Size of error message buffer
 * @return STANRUN_SUCCESS on success, error code on failure
 */
STANRUN_API int stosh_run_samplers(void* handle_ptr,
                                   const char* const* keys,
                                   const char* const* values,
                                   int num_params,
                                   char* output_dir,
                                   size_t output_dir_size,
                                   char* error_message,
                                   size_t error_message_size);

/* Free a model handle and associated resources
 * 
 * @param handle_ptr Handle returned from stosh_load_model
 */
STANRUN_API void stosh_free_model(void* handle_ptr);

/* Get the name of a loaded model
 * 
 * @param handle_ptr Handle returned from stosh_load_model
 * @return Model name string, or NULL if handle is invalid
 */
STANRUN_API const char* stosh_get_model_name(void* handle_ptr);

#ifdef __cplusplus
}
#endif

#endif
