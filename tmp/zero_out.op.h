#include "tensorflow/core/framework/op_kernel.h"

using namespace std;

/*!
 * To create one of these kernels, create a class that extends OpKernel 
 * and overrides the Compute method. The Compute method provides one 
 * context argument of type OpKernelContext*, from which you can access 
 * useful things like the input and output tensors.
 *
 * Important note: Instances of your OpKernel may be accessed concurrently. 
 * Your Compute method must be thread-safe. Guard any access to class 
 * members with a mutex (Or better yet, don't share state via class members! 
 * Consider using a ResourceMgr to keep track of Op state).
 */
class ZeroOutOp : public OpKernel {
  public:
    explicit ZeroOutOp(OpKernelConstruction *context)
      : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
      // Get the input tensor
      const Tensor &input_tensor = context->input(0);
      auto input = input_tensor.flat<int32>();

      // Create an output tensor
      Tensor*output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
            &output_tensor));
    }
};
