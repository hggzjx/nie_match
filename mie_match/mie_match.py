#encoding=utf8
import keras
from keras.models import Model
import tensorflow as tf 
import numpy as np 
import sys
sys.path.append( '../')
from base.base_model import BaseModel
from base.base_module import SoftAttention


class mie_match_1(BaseModel):
    
    def build(self):
        """
        Build the model.
        """
        a, b = self._make_inputs()

        # ---------- Embedding layer ---------- #
        embedding = self.make_embedding_layer()
        embedded_a = embedding(a)
        embedded_b = embedding(b)

        # ---------- Encoding layer ---------- #
        # encoded_a = keras.layers.Bidirectional(keras.layers.LSTM(
        #     self._params['lstm_units'],
        #     return_sequences=True,
        #     dropout=self._params['dropout_rate']
        # ))(embedded_a)
        # encoded_b = keras.layers.Bidirectional(keras.layers.LSTM(
        #     self._params['lstm_units'],
        #     return_sequences=True,
        #     dropout=self._params['dropout_rate']
        # ))(embedded_b)

        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(
                    self._params['lstm_units'],
                    return_sequences=True,
                    dropout=self._params['dropout_rate']
                ))

        encoded_a = bilstm(embedded_a)
        encoded_b = bilstm(embedded_b)

        atten_a, atten_b = SoftAttention()([encoded_a, encoded_b])

        sub_a_atten = keras.layers.Lambda(lambda x: x[0]-x[1])([encoded_a, atten_a])
        sub_b_atten = keras.layers.Lambda(lambda x: x[0]-x[1])([encoded_b, atten_b])

        mul_a_atten = keras.layers.Lambda(lambda x: x[0]*x[1])([encoded_a, atten_a])
        mul_b_atten = keras.layers.Lambda(lambda x: x[0]*x[1])([encoded_b, atten_b])

        m_a = keras.layers.concatenate([encoded_a, atten_a, sub_a_atten, mul_a_atten])
        m_b = keras.layers.concatenate([encoded_b, atten_b, sub_b_atten, mul_b_atten])

        composition_a = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(m_a)

        avg_pool_a = keras.layers.GlobalAveragePooling1D()(composition_a)
        max_pool_a = keras.layers.GlobalMaxPooling1D()(composition_a)

        composition_b = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(m_b)

        avg_pool_b = keras.layers.GlobalAveragePooling1D()(composition_b)
        max_pool_b = keras.layers.GlobalMaxPooling1D()(composition_b)

        pooled = keras.layers.concatenate([avg_pool_a, max_pool_a, avg_pool_b, max_pool_b])
        pooled = keras.layers.Dropout(rate=self._params['dropout_rate'])(pooled)

        mlp = self._make_multi_layer_perceptron_layer()(pooled)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)

        prediction = self._make_output_layer()(mlp)

        model = Model(inputs=[a, b], outputs=prediction)

        return model


class mie_match_2(BaseModel):

    def build(self):
        """
        Build the model.
        """
        a, b = self._make_inputs()

        # ---------- Embedding layer ---------- #
        embedding = self.make_embedding_layer()
        embedded_a = embedding(a)
        embedded_b = embedding(b)


        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(
                    self._params['lstm_units'],
                    return_sequences=True,
                    dropout=self._params['dropout_rate']
                ))

        encoded_a = bilstm(embedded_a)
        encoded_b = bilstm(embedded_b)

        
        
        # ---------- Local inference layer ---------- #
        atten_a, atten_b = SoftAttention()([encoded_a, encoded_b])

        sub_ab_encoded = keras.layers.Lambda(lambda x: x[0]-x[1])([encoded_a, encoded_b])
        sub_ab_atten = keras.layers.Lambda(lambda x: x[0]-x[1])([atten_a, atten_b])

        mul_ab_encoded = keras.layers.Lambda(lambda x: x[0]*x[1])([encoded_a, encoded_b])
        mul_ab_atten = keras.layers.Lambda(lambda x: x[0]*x[1])([atten_a, atten_b])

        m_atten = keras.layers.concatenate([atten_a, atten_b, sub_ab_atten, mul_ab_atten])
        m_encoded = keras.layers.concatenate([encoded_a, encoded_b, sub_ab_encoded, mul_ab_encoded])
        
        sub_ab0 = keras.layers.Lambda(lambda x: x[0]-x[1])([m_atten, m_encoded])
        mu_ab0 = keras.layers.Lambda(lambda x: x[0]*x[1])([m_atten, m_encoded])
        
        m_atten = keras.layers.concatenate([m_atten, sub_ab0, mu_ab0])
        m_encoded = keras.layers.concatenate([m_encoded, sub_ab0, mu_ab0])
        # ---------- Inference composition layer ---------- #
        composition_atten = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(m_atten)

        # avg_pool_a = keras.layers.GlobalAveragePooling1D()(composition_a)
        # max_pool_a = keras.layers.GlobalMaxPooling1D()(composition_a)

        composition_encoded = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(m_encoded)

        avg_pool_atten = keras.layers.GlobalAveragePooling1D()(composition_atten)
        max_pool_atten = keras.layers.GlobalMaxPooling1D()(composition_atten)       
        avg_pool_encoded = keras.layers.GlobalAveragePooling1D()(composition_encoded)
        max_pool_encoded = keras.layers.GlobalMaxPooling1D()(composition_encoded)

        pooled = keras.layers.concatenate([avg_pool_atten, max_pool_atten, avg_pool_encoded, max_pool_encoded])
        pooled = keras.layers.Dropout(rate=self._params['dropout_rate'])(pooled)

        # ---------- Classification layer ---------- #
        mlp = self._make_multi_layer_perceptron_layer()(pooled)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)

        prediction = self._make_output_layer()(mlp)

        model = Model(inputs=[a, b], outputs=prediction)
        # ---------- Classification layer ---------- #
        mlp = self._make_multi_layer_perceptron_layer()(pooled)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)

        prediction = self._make_output_layer()(mlp)

        model = Model(inputs=[a, b], outputs=prediction)

        return model
        





        
