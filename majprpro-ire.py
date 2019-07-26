def create_model(input_dim=300, lstm_op_dim=50, vocab_size=4000, embed_dim=300, max_len=1000, coherence_width=50, k=6,
                 start=3, model_type='tensor'):
    if (model_type == 'tensor'):
        inputs = Input(shape=(max_len, input_dim))
        lstm = LSTM(lstm_op_dim, return_sequences=True)(inputs)
        bilinear_products = Neural_Tensor_layer(output_dim=k, input_dim=lstm_op_dim)
        pairs = [((start + i * coherence_width) % max_len, (start + i * coherence_width + coherence_width) % max_len)
                 for i in range(int(max_len / coherence_width))]
        similarity_pairs = [(Lambda(lambda t: t[:, p[0], :])(lstm), Lambda(lambda t: t[:, p[1], :])(lstm)) for p in
                            pairs]
        sigmoid_layer = Dense(1, activation="sigmoid")
        similarities = [sigmoid_layer(bilinear_products([w[0], w[1]])) for w in similarity_pairs]
        tmp = Temporal_Mean_Pooling()(lstm)
        simi = Concatenate()([i for i in similarities])
        tmp_simi = Concatenate()([tmp, simi])
        dense1 = Dense(256, activation='relu')(tmp_simi)
        dense2 = Dense(64, activation='relu')(dense1)
        out = Dense(1, activation='linear')(dense2)
        model = Model(inputs=inputs, outputs=out)
    elif (model_type == 'lstm'):
        inputs = Input(shape=(max_len, input_dim))
        lstm = LSTM(lstm_op_dim, return_sequences=False)(inputs)
        op = Dense(1, activation='linear')(lstm)
        model = Model(inputs=inputs, outputs=op)
    return model


class Temporal_Mean_Pooling(Layer):
    def __init__(self, **kwargs):
        self.input_spec = InputSpec(ndim=3)
        super(Temporal_Mean_Pooling, self).__init__(**kwargs)

    def call(self, x):
        mask = K.mean(K.ones_like(x), axis=-1)
        return K.sum(x, axis=-2) / K.sum(mask, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class Neural_Tensor_layer(Layer):
    def __init__(self, output_dim, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Neural_Tensor_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        mean = 0.0
        std = 1.0
        k = self.output_dim
        d = self.input_dim
        W = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k, d, d))
        V = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2 * d, k))
        self.W = K.variable(W)
        self.V = K.variable(V)
        self.b = K.zeros((self.input_dim,))
        self.trainable_weights = [self.W, self.V, self.b]

    def call(self, inputs, mask=None):
        e1 = inputs[0]
        e2 = inputs[1]
        batch_size = K.shape(e1)[0]
        k = self.output_dim
        feed_forward = K.dot(K.concatenate([e1, e2]), self.V)
        bilinear_tensor_products = [K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1)]
        for i in range(k)[1:]:
            btp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
            bilinear_tensor_products.append(btp)
        result = K.tanh(K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward)
        return result

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, self.output_dim)