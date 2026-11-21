"""
nids_layers.py
==============
Custom Keras layer used by the Feature Transformer model.

Keep this file in the same folder as the notebooks. Both `adv_dl_models.ipynb` (which trains the
model) and `ensemble_decision.ipynb` (which loads it) import it, because Keras needs the class to be
registered before it can load a saved model that uses it.
"""

import keras


@keras.saving.register_keras_serializable(package="nids")
class FeatureTokenizer(keras.layers.Layer):
    """
    Turns a row of n numeric features into n 'tokens' of size d_model.
    Each feature gets its own learned weight vector and bias, so token i is feature i's value
    times w_i plus b_i. This gives the attention layers a separate embedding per feature
    (the idea behind the FT-Transformer for tabular data).
    """

    def __init__(self, n_features, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.d_model = d_model

    def build(self, input_shape):
        self.w = self.add_weight(name="w", shape=(self.n_features, self.d_model),
                                 initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.n_features, self.d_model),
                                 initializer="zeros")

    def call(self, x):
        # (batch, n) -> (batch, n, 1) * (n, d) + (n, d) -> (batch, n, d)
        return keras.ops.expand_dims(x, -1) * self.w + self.b

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_features, self.d_model)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_features": self.n_features, "d_model": self.d_model})
        return cfg
