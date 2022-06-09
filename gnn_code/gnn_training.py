import math
import os
import warnings
from datetime import datetime
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import HTML, display
from keras import layers
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.python.util import deprecation

# Contrôle des alertes dues à la version de Python utlisée
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:90% !important; }</style>"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0


# results directory
dst = "results"


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        #         fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.relu))
    #         fnn_layers.append(layers.Dense(units, activation=tf.nn.leaky_relu))
    #         fnn_layers.append(layers.Dense(units, activation=tf.nn.selu))

    return tf.keras.Sequential(fnn_layers, name=name)


class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize
        self.hidden_units = hidden_units
        self.ffn_prepare = create_ffn(self.hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        num_segments = tf.math.reduce_max(node_indices) + 1

        num_rows = tf.shape(node_indices)[0]
        rows_idx = tf.range(num_rows)
        segment_ids_per_row = node_indices + num_segments * tf.expand_dims(
            rows_idx, axis=1
        )

        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages,
                segment_ids_per_row,
                num_segments=num_segments * num_rows,
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages,
                segment_ids_per_row,
                num_segments=num_segments * num_rows,
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages,
                segment_ids_per_row,
                num_segments=num_segments * num_rows,
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        aggregated_message = tf.reshape(
            aggregated_message, [num_rows, num_segments, self.hidden_units[-1]]
        )
        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        #         print(node_repesentations)

        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=1)
        return node_embeddings

    def call(
        self, node_repesentations, node_indices, neighbour_indices, edge_weights=None
    ):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges,
        edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        neighbour_repesentations = tf.gather(
            node_repesentations, neighbour_indices, axis=1, batch_dims=1
        )
        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        #         print(neighbour_messages)

        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(node_indices, neighbour_messages)
        #         print(aggregated_messages)

        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)


class GraphSequence(tf.keras.utils.Sequence):
    def __init__(self, X, na, nb, mask, y, batch_size):
        self.X = X
        self.na = na
        self.nb = nb
        self.mask = mask
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return [
            self.X[start:end],
            self.na[start:end],
            self.nb[start:end],
            self.mask[start:end],
        ], self.y[start:end]


def start_trail():

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"{dst}/{stamp}/summary.csv"

    # Parameters for the model
    mask_type = "unique"
    num_epochs = 128
    hidden_units = (128, 128, 128)
    aggregation_type = "mean"
    combination_type = "add"
    dropout_rate = 0.10
    normalize = False
    n_gcl = 6  # number of GraphConvLayer
    batch_size = 128
    learning_rate = 1e-4

    # Number of inputs
    ll = 350  # maximum number of nodes
    el = 850  # maximum number of edges
    vl = 183  # length of the feature vector

    output_text = f"mask_type,{mask_type}\n"
    output_text += f"num_epochs,{num_epochs}\n"
    output_text += f"hidden_units,{len(hidden_units)}*{hidden_units[0]}\n"
    output_text += f"aggregation_type,{aggregation_type}\n"
    output_text += f"combination_type,{combination_type}\n"
    output_text += f"dropout_rate,{dropout_rate}\n"
    output_text += f"normalize,{normalize}\n"
    output_text += f"GraphConvLayer,{n_gcl}\n"

    print("======Parameters======")
    print(output_text)
    print("======Trail Starts======")

    # construct input layers
    a_input = tf.keras.Input(shape=(ll, vl), dtype=tf.float32)
    mask_input = tf.keras.Input(shape=(ll, 1))
    ei_input = tf.keras.Input(shape=(el), dtype=tf.int32)
    eo_input = tf.keras.Input(shape=(el), dtype=tf.int32)

    preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
    x = preprocess(a_input)

    for i in range(1, n_gcl + 1):
        # create GraphConvLayer
        gcl = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name=f"graph_conv{i}",
        )
        # apply layer and skip connection
        x = gcl(x, ei_input, eo_input) + x

    postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
    compute_logits = layers.Dense(units=1, name="logits")
    # Fetch node embeddings for the input node_indices.
    node_embeddings = postprocess(x)
    # Compute logits
    y = compute_logits(node_embeddings)
    y = tf.keras.layers.Multiply()([y, mask_input])
    gnn = tf.keras.Model(inputs=[a_input, ei_input, eo_input, mask_input], outputs=y)

    # compile model
    gnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="acc")],
    )

    # load train data
    ai = np.load("features/fixed_train_X.npy")
    ei = np.load("features/fixed_train_na.npy")
    eo = np.load("features/fixed_train_nb.npy")
    mask = np.load(f"features/combined_train_{mask_type}.npy")
    ao = np.load("features/scaled_train_y.npy")
    # load dev data
    dai = np.load("features/fixed_dev_X.npy")
    dei = np.load("features/fixed_dev_na.npy")
    deo = np.load("features/fixed_dev_nb.npy")
    dmask = np.load(f"features/combined_dev_{mask_type}.npy")
    dao = np.load("features/scaled_dev_y.npy")

    # make training batches
    train_seq = GraphSequence(ai, ei, eo, mask, ao, batch_size)
    dev_seq = GraphSequence(dai, dei, deo, dmask, dao, batch_size)

    # starts training
    train_start = perf_counter()
    history = gnn.fit(
        x=train_seq,
        epochs=num_epochs,
        validation_data=dev_seq,
    )
    train_time = (perf_counter() - train_start) / 3600
    output_text += f"train_time,{train_time}_hrs\n"

    gnn.save(f"{dst}/{stamp}")

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "dev"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "dev"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.savefig(f"{dst}/{stamp}/plot.png")

    # load test data
    tai = np.load("features/fixed_test_X.npy")
    tei = np.load("features/fixed_test_na.npy")
    teo = np.load("features/fixed_test_nb.npy")
    # tmask = np.load(f"features/combined_test_{mask_type}.npy")
    tmask = np.load("features/combined_test_pad.npy")
    tao = np.load("features/scaled_test_y.npy")
    tlabels = np.load("features/combined_test_an.npy")
    test_seq = GraphSequence(tai, tei, teo, tmask, tao, batch_size)
    pred = np.squeeze(gnn.predict(test_seq))

    scaler = joblib.load("scaler_y.joblib")
    y_pred_overall = scaler.inverse_transform(pred[tmask].reshape(-1, 1))
    y_true_overall = scaler.inverse_transform(tao[tmask].reshape(-1, 1))
    mae_overal = mae(y_pred_overall, y_true_overall)
    output_text += f"mae_overal,{mae_overal}\n"

    for atomic_number in range(1, 119):
        idx = tlabels == atomic_number
        if idx.any():
            y_pred = scaler.inverse_transform(pred[idx].reshape(-1, 1))
            y_true = scaler.inverse_transform(tao[idx].reshape(-1, 1))
            output_text += f"mae_{atomic_number},{mae(y_true, y_pred)}\n"

    with open(output_csv, "w") as wf:
        wf.write(output_text)
        wf.flush()


if __name__ == "__main__":
    start_trail()
