

#include <unistd.h>

class Output {

  virtual size_t register_block() = 0;

  virtual void *get_landing_pointer() = 0;

  virtual void register_transfer_complete() = 0;
};
