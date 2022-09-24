import onnx

# Use some symbolic name not used for any other dimension
# or an actal value
new_batch_dim = 16
in_onnx_path = "onnx/shufflenet_v1.onnx"
out_onnx_path = "onnx/shufflenet_v1.16.onnx"
input_name = 'gpu_0/data_0'


def change_input_dim(model):
    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim
    inputs = model.graph.input
    for input in inputs:
        if input.name != input_name:
            continue
        print(input)
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        # dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        dim1.dim_value = new_batch_dim


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)


apply(change_input_dim, in_onnx_path, out_onnx_path)
