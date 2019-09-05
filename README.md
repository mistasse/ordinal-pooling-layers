# Ordinal Pooling Layer

A Keras implementation of the Ordinal Pooling layer proposed in the [Ordinal Pooling paper](https://orbi.uliege.be/handle/2268/238475). `OrdinalPooling2D` and `GlobalOrdinalPooling2D` can be found in the `ordinal_pooling.py` file.

At the time of this implementation, tensorflow's *sort* function appeared to run on the CPU, hence the min-max swapping sort to keep it on the GPU for local poolings.

