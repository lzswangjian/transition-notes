#include <string>

class ParserReader : public OpKernel {
  public:
    explicit ParserReader(OpKernelConstruction *context) : OpKernel(context) {

    }

  private:
    TaskContext task_context_;

    string arg_prefix_;

    // mutex to synchronize access to Compute.
    mutex mu_;

    // How many times the document source has been rewinded.
    int num_epochs_ = 0;
    
    // How many sentences this op can be processing at any given time.
    int max_batch_size_ = 1;

    int feature_size_;
};
