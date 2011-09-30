#ifndef __MEMORY__STACK_ALLOC__H__
#define __MEMORY__STACK_ALLOC__H__

#include <stdlib.h>
#include <vector>

namespace stan { 

  namespace memory { 

    namespace {
      const size_t DEFAULT_INITIAL_NBYTES = 1 << 10;
    }

    /**
     * Here's an example of how to use HMC for a simple model with
     * strong parameter correlations.
     *
     * @example bivar_norm.cpp  Simple example of full HMC sampling.
     */


    /**
     * An instance of this class provides a memory pool through
     * which blocks of raw memory may be allocated and then collected
     * simultaneously.
     * 
     * This class is useful in settings where large numbers of small
     * objects are allocated and then collected all at once.  This may
     * include objects whose destructors have no effect.
     * 
     * Memory is allocated on a stack of blocks.  Each block allocated
     * is twice as large as the previous one.  The memory may be
     * recovered, with the blocks being reused, or all blocks may be
     * freed, resetting the stack of blocks to its original state. 
     */
    class stack_alloc {
    private: 
      std::vector<char*> blocks_;
      std::vector<size_t> sizes_; // could store initial & shift for others
      unsigned int cur_block_;
      size_t used_;
    public:

      /**
       * Construct a resizable stack allocator initially holding the
       * default number of bytes, 1024.
       */
      stack_alloc() :
	blocks_(1, static_cast<char*>(malloc(DEFAULT_INITIAL_NBYTES))),
	sizes_(1,DEFAULT_INITIAL_NBYTES),
	cur_block_(0),
	used_(0) {
      }

      /**
       * Construct a resizable stack allocator initially holding the
       * specified number of bytes.
       *
       * @param $initial_nbytes Initial number of bytes for the
       * allocator.
       */
      stack_alloc(size_t initial_nbytes) :
	blocks_(1, static_cast<char*>(malloc(initial_nbytes))),
	sizes_(1,initial_nbytes),
	cur_block_(0),
	used_(0) {
      }

      /**
       * Destroy this memory allocator.
       *
       * This is implemented as a no-op as there is no destruction
       * required.
       */
      ~stack_alloc() { 
      }

      /**
       * Return a newly allocated block of memory of the appropriate
       * size managed by the stack allocator.
       *
       * This function may call C++'s <code>malloc()</code> function,
       * with any exceptions percolated throught this function.
       *
       * @param size_t $len Number of bytes to allocate.
       * @return A pointer to the allocated memory.
       */
      inline void* alloc(size_t len) {
	if (sizes_[cur_block_] < used_ + len)
	  ++cur_block_;
 
	while (cur_block_ < blocks_.size() && sizes_[cur_block_] < len)
	  ++cur_block_;
		   
	if (cur_block_ >= sizes_.size()) {
	  size_t newsize = sizes_.back() * 2;
	  if (newsize < len)
	    newsize = len;
	  blocks_.push_back((char*)malloc(newsize));
	  sizes_.push_back(newsize);
	  used_ = 0;
	}

	void* result = &blocks_[cur_block_][used_];
	used_ += len;
	return result;
      }

      /**
       * Recover all the memory used by the stack allocator.  The stack
       * of memory blocks allocated so far will be available for further
       * allocations.  To free memory back to the system, use the
       * function free_all().
       */
      inline void recover_all() {
	cur_block_ = 0;
	used_ = 0;
      }
    
      /**
       * Free all memory used by the stack allocator back to the system.
       */
      inline void free_all() {
	for (unsigned int i = 0; i < cur_block_; ++i)
	  free(blocks_[i]);
	recover_all();
      }
  
    };

  }
}
#endif
