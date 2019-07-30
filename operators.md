# Interacting with machin learning models

Operators are one of the fundamental buidling blocks to interact with machine learning models on the Cortex network. A operator is a set of statements that performs a task or calculates a value. To use a function, you must define it somewhere in the scope from which you wish to call it.

## CVM Operators

**nn**

| OP         | Name          | Requirement       | Default          |
|------------|---------------|-------------------|------------------|
| Conv2D     | channels      |                   |                  |
|            | kernel_size   | 2-D               |                  |
|            | strides       | 2-D               | (1, 1)           |
|            | padding       | 2-D               | (0, 0)           |
|            | dilation      | 2-D               | (1, 1)           |
|            | groups        | 1 or in_channels  | 1                |
|            | layout        | NCHW              | NCHW             |
|            | kernel_layout | OIHW              | OIHW             |
|            | out_layout    | __undef__ or NCHW | __undef__        |
|            | out_dtype     | -1 or 4(kInt32)   | -1               |
|            | use_bias      |                   | TRUE             |
|            | input data    | <= INT8           |                  |
| Dense      | units         | >=1               |                  |
|            | input data    | <= INT8           |                  |
| Upsampling | use_bias      |                   | TRUE             |
|            | scale         | >0                |                  |
|            | layout        | NCHW              | NCHW             |
|            | method        | NEAREST_NEIGHBOR  | NEAREST_NEIGHBOR |
| MaxPool2D  | pool_size     | 2-D               |                  |
|            | strides       | 2-D               | (1, 1)           |
|            | padding       | 1-D or 2-D        | (0, 0)           |
|            | layout        | NCHW              | NCHW             |
|            | ceil_mode     | FALSE             | FALSE            |

**reduce**

| OP     | Name     | Requirement          | Default |
|--------|----------|----------------------|---------|
| sum,max | axis     | -ndim <= axis < ndim | ()      |
|        | keepdims |                      | FALSE   |
|        | exclude  |                      | FALSE   |
|        | dtype    | kInt32               | kInt32  |

**transform**

| OP            | Name        | Requirement                    | Default |
|---------------|-------------|--------------------------------|---------|
| expand_dims   | axis        | -ndim-1 <= axis <= ndim        |         |
|               | num_newaxis | >=0                            | 1       |
| transpose     | axes        | NONE or =ndim                  | NONE    |
| reshape       | shape       |                                |         |
| squeeze       | axis        | -ndim <= axis < ndim           | NONE    |
| concatenate   | axis        | -ndim <= axis < ndim           | 1       |
| take          | axis        | -ndim <= axis < ndim           | NONE    |
| strided_slice | begin       |                                |         |
|               | end         |                                |         |
|               | stride      |                                |         |
| repeat        | repeats     | >=1                            |         |
|               | axis        | -ndim <= axis < ndim           | 0       |
| tile          | reps        |                                | NONE    |
| slice_like    | axis        | -src_ndim <= axis <= dest_ndim | NONE    |
| cvm_lut       |             |                                |         |
| flatten       |             |                                |         |

**vision**

| OP                  | Name              | Requirement             | Default |
|---------------------|-------------------|-------------------------|---------|
| get_valid_count     | score_threshold   |                         | 0       |
| non_max_suppression | return_indices    | FALSE                   | FALSE   |
|                     | iou_threshold     | multiply 100 by default | 50      |
|                     | force_suppress    |                         | FALSE   |
|                     | top_k             |                         | -1      |
|                     | id_index          |                         | 0       |
|                     | coord_start       |                         | 2       |
|                     | score_index       |                         | 1       |
|                     | max_output_size   |                         | -1      |
|                     | invalid_to_bottom | TRUE                    | TRUE    |

**broadcast**

| OP            | Name | Requirement | Default |
|---------------|------|-------------|---------|
| broadcast_add |      |             |         |
| broadcast_sub |      |             |         |
| broadcast_mul |      |             |         |
| broadcast_max |      |             |         |

**elemwise**

| OP              | Name | Requirement | Default |
|-----------------|------|-------------|---------|
| abs             |      |             |         |
| log2            |      |             |         |
| elemwise_add    |      |             |         |
| elemwise_sub    |      |             |         |
| negative        |      |             |         |
| clip            |      |             |         |
| cvm_clip        |      |             |         |
| cvm_right_shift |      |             |         |
| cvm_left_shift  |      |             |         |

## CVM Operator Attributes

**nn/convolution.cc**

| Name   | Precision Check | Attribute Check |
| ------ | --------------- | --------------- |
| conv2d | ✓               | ✓               |

**nn/nn.cc**

| Name   | Precision Check | Attribute Check |
|--------|-----------------|-----------------|
| conv2d | ✓               | ✓               |
| dense  | ✓               | ✓               |
| relu   | ✓               | ✓               |

**nn/pooling.cc**

| Name       | Precision Check | Attribute Check |
| ---------- | --------------- | --------------- |
| max_pool2d | ✓               | ✓               |

**nn/upsampling.cc**

| Name       | Precision Check | Attribute Check |
| ---------- | --------------- | --------------- |
| upsampling | ✓               | ✓               |

**nn/nms.cc**

| Name                | Precision Check | Attribute Check |
|---------------------|-----------------|-----------------|
| non_max_suppression | ✓               | ✓               |
| get_valid_counts    | ✓               | ✓               |

**tensor/broadcast.cc**

| Name                | Precision Check | Attribute Check |
|---------------------|-----------------|-----------------|
| broadcast_add       | ✓               | ✓               |
| broadcast_sub       | ✓               | ✓               |
| broadcast_mul       | ✓               | ✓               |
| broadcast_max       | ✓               | ✓               |

**tensor/reduce.cc**

| Name   | Precision Check | Attribute Check |
|--------|-----------------|-----------------|
| sum    | ✓               | ✓               |
| max    | ✓               | ✓               |

**tensor/elemwise.cc**

| Name                | Precision Check | Attribute Check |
|---------------------|-----------------|-----------------|
| abs                 | ✓               | ✓               |
| log2                | ✓               | ✓               |
| elemwise_add        | ✓               | ✓               |
| elemwise_sub        | ✓               | ✓               |
| negative            | ✓               | ✓               |
| clip                | ✓               | ✓               |
| cvm_clip            | ✓               | ✓               |
| cvm_right_shift     | ✓               | ✓               |
| cvm_left_shift      | ✓               | ✓               |

**tensor/transform.cc**

| Name                | Precision Check | Attribute Check |
|---------------------|-----------------|-----------------|
| repeat              | ✓               | ✓               |
| tile                | ✓               | ✓               |
| flatten             | ✓               | ✓               |
| concatenate         | ✓               | ✓               |
| expand_dims         | ✓               | ✓               |
| reshape             | ✓               | ✓               |
| squeeze             | ✓               | ✓               |
| transpose           | ✓               | ✓               |
| slice\|strided_slice |✓ | ✓               |
| take                | ✓               | ✓               |
| cvm_lut             | ✓               | ✓               |
| slice_like          | ✓               | ✓               |