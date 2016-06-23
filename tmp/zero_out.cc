#include "tensorflow/core/fraework/op.h"

/*!
 * The name of the Op should be unique and CamelCase. 
 * Names starting with an underscore (_) are reserved for internal use.
 */
REGISTER_OP("ZeroOut")
  .Input("to_zero: int32")
  .Output("zeroed: int32");


// Register
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
