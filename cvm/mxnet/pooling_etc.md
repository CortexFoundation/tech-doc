#### Pooling

*pool_type*: indicates pooling type, only supported `avg` and `max`.
*is_global*: indicates is global average pooling.
*pooling_convention*: must be `valid` for equivalent transformation with depthwise *Convolution*.
*count_include_pad*: must be `true` for equivalent transformation with depthwise *Convolution*.
*kernel, stride, pad*: pooling kernel, stride and pad attributes.

##### Rewrite GlobalAvgPooling 

```python
GlobalAvgPooling(data) =
```

$$
\sum_{k_i, k_j}^{kernel}data[:,:,k_i,k_j] / (size_{kernel})
$$

```python
= broadcast_mul(sum(data, axis=(2, 3)), scale), which scale equals 1 / K / K
```

##### Rewrite AvgPooling

```python
AvgPooling(data) = Convolution(data,
            kernel=kernel, stride=stride, pad=pad, # depthwise conv2d
            no_bias=True, dilate=(1, 1), layout='NCHW',
            num_filter=in_channel, num_group=in_channel)
```

#### Rewrite LeakyReLU

*act_type*: action type, only supported `leaky`.
*slope*: attribute.

```python
LeakyReLU(data) = relu(data) - slope * relu(-data)
```


#### Rewrite Dropout

do nothing in inference and strip it.

#### Rewrite _div_scalar, _mul_scalar

To avoid division in INT8 graph, we use the operator `broadcast_mul` to rewrite the scalar operator above.




```python
scale = alpha / clip ie. target_range = alpha / scale = clip
we calculate shift_bit with sb = ceil(log2(scale)) 
ie. scale <= 2 ** (sb) < scale * 2
so that we can get target range: 
    target_range = alpha / (2 ** sb) <= alpha / scale = clip
  target_range = alpha / (2 ** sb) > alpha / (scale * 2) = clip / 2 
```

So the target range does not span full space of INT `bit`, its max value is between $(clip/2, clip]$. Usually, the target range decreases and causes accuracy after quantization to be lower. But it does reduce the work of realizing the requantization operator for the scale in the next steps.