import tensorflow as tf

class IdBlock(tf.keras.Model):
    def __init__(self , filters=64, kernel_size=3, activation='gelu', name=''):
        super(IdBlock , self ).__init__(name=name)
        self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', strides=2)
        self.norm = tf.keras.layers.BatchNormalization()
        # self.norm = tf.keras.layers.LayerNormalization()
        self.act  = tf.keras.layers.Activation(activation)
        self.concat  = tf.keras.layers.Concatenate()
    def call(self , input):
        x = self.conv(input)
        x = self.norm(x, training=False)
        x = self.act(x)
        

        x = self.conv(x)
        x = self.norm(x, training=False)

        x = self.concat([x , input])
        x = self.act(x)
        return x
    
class PreBlock(tf.keras.Model):
    def __init__(self , filters=64, kernel_size=7, pool_size=3, activation='gelu', name=''):
        super(PreBlock , self).__init__(name="")
        self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', strides=2)
        self.norm  = tf.keras.layers.BatchNormalization()
        # self.norm = tf.keras.layers.LayerNormalization()
        self.act  = tf.keras.layers.Activation(activation)
        self.pool  = tf.keras.layers.MaxPool2D(pool_size=(1, pool_size), padding='same', strides=1)

    def call(self , input):
        x = self.conv(input)
        x = self.norm(x, training=False)
        x = self.act(x)
        x = self.pool(x)

        return x
    
class IdBranch(tf.keras.Model):
    def __init__(self , preblock_filters=64,
            preblock_kernel_size=7,
            preblock_pool_size=3,
            idblock_kernel_sizes=3,
            idblock_filters = [64, 128],
            idblock_activation='gelu',
                 idblock_avg_pool_size=2,
                ):
        super(IdBranch , self).__init__(name="")
        self.preblock = PreBlock(filters=preblock_filters, kernel_size=preblock_kernel_size, pool_size=preblock_pool_size, name='')
        self.idbls = []
        for i, idblock_filter in enumerate(idblock_filters):
            self.idbls.append(IdBlock(idblock_filter, kernel_size=idblock_kernel_sizes, activation=idblock_activation))
        self.apool = tf.keras.layers.AveragePooling1D(pool_size=idblock_avg_pool_size, padding='SAME')
        # self.gpool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self , input):
        x = self.preblock(input)
        for i, idbl in enumerate(self.idbls):
            x = idbl(x)
        x = self.apool(x)
        # x = self.gpool(x)

        return x
    
class ResBranch(tf.keras.Model):
    def __init__(self , preblock_filters=64,
                preblock_kernel_size=7,
                preblock_pool_size=3,
                idblock_kernel_sizes=[3, 5, 7],
                idblock_filters = [64, 128, 256],
                idblock_activation='gelu',
                idblock_avg_pool_sizes=[16, 11, 6],
                ):
        super(ResBranch , self).__init__(name="")
        self.preblock = PreBlock(filters=preblock_filters, kernel_size=preblock_kernel_size, pool_size=preblock_pool_size, name='')
        self.kernel_idbls = []
        for i, idblock_kernel_size in enumerate(idblock_kernel_sizes):
            self.kernel_idbls.append(
                IdBranch(preblock_filters=preblock_filters,
                                    preblock_kernel_size=preblock_kernel_size,
                                    preblock_pool_size=preblock_pool_size,
                                    idblock_kernel_sizes=idblock_kernel_size,
                                    idblock_filters = idblock_filters,
                                    idblock_activation=idblock_activation,
                                     idblock_avg_pool_size=idblock_avg_pool_sizes[i],
                                             )
                                    )

    def call(self , input):
        x = self.preblock(input)
        out = []
        for i, kernel_idbl in enumerate(self.kernel_idbls):
            out.append(kernel_idbl(x))


        return out
    
class COSMELayer(tf.keras.Model):
    def __init__(self , preblock_filters=64,
                preblock_kernel_sizes=[37],
                preblock_pool_size=3,
                idblock_kernel_sizes=[3, 5, 7],
                idblock_filters = [64, 128, 256],
                idblock_activation='gelu',
                idblock_avg_pool_sizes=[16, 11, 6],
                last_activation='softmax',
                n_classes=10,
                ):
        super(COSMELayer , self).__init__(name="COSMELayer")
        self.kernel_preblocks = []
        for i, preblock_kernel_size in enumerate(preblock_kernel_sizes):
            self.kernel_preblocks.append(
                ResBranch(
                    preblock_filters=preblock_filters,
                    preblock_kernel_size=preblock_kernel_size,
                    preblock_pool_size=preblock_pool_size,
                    idblock_kernel_sizes=idblock_kernel_sizes,
                    idblock_filters = idblock_filters,
                    idblock_activation=idblock_activation,
                    idblock_avg_pool_sizes=idblock_avg_pool_sizes,
                                             )
                                    )
            
        self.concat = tf.keras.layers.Concatenate(axis=1)
        # self.gpool = tf.keras.layers.GlobalAveragePooling1D()
        # self.norm  = tf.keras.layers.Dropout(0.1)
        self.flatten = tf.keras.layers.Flatten()
        # self.norm = tf.keras.layers.BatchNormalization()
        # self.tanh = tf.keras.layers.Dense(n_classes, activation='tanh')
        # self.dense = tf.keras.layers.Dense(2048, activation='relu')
        # self.leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        self.classifier = tf.keras.layers.Dense(n_classes, activation=last_activation)

    def call(self , input):
        # x = self.preblock(input)
        out = []
        for i, kernel_preblock in enumerate(self.kernel_preblocks):
            out.extend(kernel_preblock(input))
        out = self.concat(out)
        # out = self.gpool(out)
        out = self.flatten(out)
        # out = tf.squeeze(out)
        # out = self.tanh(out)
        # out = self.dense(out)
        # out = self.norm(out)
        # out = self.leaky_relu(out)
        # out = self.norm(out)
        out = self.classifier(out)
        # values, indices = tf.math.top_k(out, 3)
        # values = tf.cast(values, tf.float32)
        # indices = tf.cast(values, tf.float32)
        # out = self.concat([values, indices], axis=1)
        # out = self.flatten(out)
        # out = self.dense(out)
        # out = self.dense(out)
        # out = self.classifier(out)


        return out
