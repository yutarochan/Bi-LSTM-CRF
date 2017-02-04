# Bi-LSTM-CRF
Experimental implementation of various mapped many-to-many recurrent networks.
The repository includes the following different architectures:
* Vanilla LSTM Architecture
* Bidirectional LSTM Architecture
* Bidirectional LSTM + Conditional Random Fields (CRF) Architecture

### Key Implementation Notes
* The general structure of the text-to-tag model is built from the following structure:

        model = Sequence()
        model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))

        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

* Note the use of the `TimeDistributed` layer which enables the many-to-many architecture.
* This architecture will work with variable length texts.
* The only tricky part of getting this to work is fixing your data dimensions;
notably your `y_train` and `y_test` must be 3D and needs to be adjusted accordingly.
    * See https://github.com/fchollet/keras/issues/3916#issuecomment-250689482
    * See the BiLSTM Example in the repo for how to change dimension.
    * Just encapsulate each of the integer label in another list [x].
