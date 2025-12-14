"""
Graph Neural Network for SBOM Vulnerability Analysis.

Implements GNN models for:
- Node classification (vulnerability prediction per package)
- Graph classification (overall supply chain risk)
- Link prediction (dependency risk analysis)

Based on 2025 research on GNN-based vulnerability detection:
- Graph Convolutional Networks (GCN) for local feature aggregation
- Graph Attention Networks (GAT) for importance weighting
- Message Passing Neural Networks for edge-aware propagation
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None

from medtech_ai_security.sbom_analysis.graph_builder import GraphData

logger = logging.getLogger(__name__)


@dataclass
class GNNConfig:
    """Configuration for the GNN model."""

    # Model architecture
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 4  # For GAT
    dropout_rate: float = 0.2
    use_attention: bool = True

    # Input/output
    input_dim: int = 88  # NodeFeatures.feature_dim
    num_classes: int = 3  # 0: clean, 1: vulnerable, 2: transitive

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 32

    # Message passing
    aggregation: str = "mean"  # mean, sum, max
    normalize: bool = True


class TransposeLayer(keras.layers.Layer if TF_AVAILABLE else object):  # type: ignore[misc]
    """Custom layer to transpose edge_index for Keras 3.x compatibility.

    In Keras 3.x, tf.transpose cannot be called directly on KerasTensors
    in the functional API. This layer wraps the transpose operation.
    """

    def __init__(self, **kwargs: Any) -> None:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for GNN models")
        super().__init__(**kwargs)

    def call(self, inputs: Any) -> Any:
        """Transpose the input tensor."""
        return tf.transpose(inputs)

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        return config


class GraphConvLayer(keras.layers.Layer if TF_AVAILABLE else object):  # type: ignore[misc]
    """Graph Convolutional Layer (GCN).

    Implements: H' = D^(-1/2) * A * D^(-1/2) * H * W
    where A is the adjacency matrix with self-loops.
    """

    def __init__(
        self,
        output_dim: int,
        activation: str = "relu",
        use_bias: bool = True,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for GNN models")
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.activation_name = activation
        self.use_bias = use_bias
        self.normalize = normalize

    def build(self, input_shape: Any) -> None:
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None

        self.activation = keras.activations.get(self.activation_name)
        super().build(input_shape)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        node_features, edge_index, num_nodes = inputs

        # Transform features
        h = tf.matmul(node_features, self.kernel)

        # Message passing: aggregate neighbor features
        # Use tf.shape for dynamic shape check (Keras 3.x compatible)
        num_edges = tf.shape(edge_index)[1]

        def message_pass() -> Any:
            # Get source and target indices
            src = edge_index[0]
            tgt = edge_index[1]

            # Gather source features
            messages = tf.gather(h, src)

            # Aggregate messages at target nodes
            aggregated = tf.zeros((num_nodes, self.output_dim), dtype=h.dtype)
            aggregated = tf.tensor_scatter_nd_add(aggregated, tf.expand_dims(tgt, 1), messages)

            # Degree normalization
            if self.normalize:
                # Count incoming edges per node
                ones = tf.ones((tf.shape(tgt)[0],), dtype=h.dtype)
                degree = tf.zeros((num_nodes,), dtype=h.dtype)
                degree = tf.tensor_scatter_nd_add(degree, tf.expand_dims(tgt, 1), ones)
                degree = tf.maximum(degree, 1.0)  # Avoid division by zero

                # Normalize by degree
                aggregated = aggregated / tf.expand_dims(degree, 1)

            # Combine with self-loop (original features)
            return h + aggregated

        def no_edges() -> Any:
            # No edges, just use transformed features
            return h

        h = tf.cond(num_edges > 0, message_pass, no_edges)

        if self.bias is not None:
            h = h + self.bias

        return self.activation(h)

    def compute_output_shape(self, input_shape: Any) -> tuple[Any, int]:
        """Compute output shape for Keras 3.x compatibility."""
        return (input_shape[0][0], self.output_dim)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "activation": self.activation_name,
                "use_bias": self.use_bias,
                "normalize": self.normalize,
            }
        )
        result: dict[str, Any] = config
        return result


class GraphAttentionLayer(keras.layers.Layer if TF_AVAILABLE else object):  # type: ignore[misc]
    """Graph Attention Layer (GAT).

    Implements multi-head attention for graph data.
    """

    def __init__(
        self,
        output_dim: int,
        num_heads: int = 4,
        concat_heads: bool = True,
        dropout_rate: float = 0.0,
        activation: str = "elu",
        **kwargs: Any,
    ) -> None:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for GNN models")
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.activation_name = activation

    def build(self, input_shape: Any) -> None:
        input_dim = input_shape[0][-1]

        # Per-head output dimension
        if self.concat_heads:
            self.head_dim = self.output_dim // self.num_heads
        else:
            self.head_dim = self.output_dim

        # Weight matrices for each head
        self.W = self.add_weight(
            name="W",
            shape=(self.num_heads, input_dim, self.head_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Attention weights
        self.a_src = self.add_weight(
            name="a_src",
            shape=(self.num_heads, self.head_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.a_tgt = self.add_weight(
            name="a_tgt",
            shape=(self.num_heads, self.head_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.activation = keras.activations.get(self.activation_name)
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.leaky_relu = keras.layers.LeakyReLU(negative_slope=0.2)

        super().build(input_shape)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        node_features, edge_index, num_nodes = inputs

        # Transform features for each head: (num_heads, num_nodes, head_dim)
        h = tf.einsum("ni,hid->hnd", node_features, self.W)

        # Use tf.shape for dynamic shape check (Keras 3.x compatible)
        num_edges_tensor = tf.shape(edge_index)[1]

        def attention_pass() -> Any:
            src = edge_index[0]
            tgt = edge_index[1]

            # Compute attention scores
            # (num_heads, num_nodes, 1)
            attn_src = tf.einsum("hnd,hdo->hno", h, self.a_src)
            attn_tgt = tf.einsum("hnd,hdo->hno", h, self.a_tgt)

            # Gather attention scores for edges
            # (num_heads, num_edges)
            edge_attn = (
                tf.gather(attn_src, src, axis=1)[:, :, 0]
                + tf.gather(attn_tgt, tgt, axis=1)[:, :, 0]
            )
            edge_attn = self.leaky_relu(edge_attn)

            # Softmax over neighbors (per target node)
            # Create sparse attention matrix and apply softmax
            num_edges = tf.shape(src)[0]
            edge_weights = tf.nn.softmax(edge_attn, axis=1)

            # Note: dropout applied unconditionally in graph mode, controlled by training flag
            edge_weights = self.dropout(edge_weights, training=training)

            # Aggregate messages with attention weights
            src_features = tf.gather(h, src, axis=1)  # (num_heads, num_edges, head_dim)
            weighted_messages = src_features * tf.expand_dims(edge_weights, -1)

            # Scatter to target nodes
            output = tf.zeros((self.num_heads, num_nodes, self.head_dim), dtype=h.dtype)
            tgt_expanded = tf.tile(tf.expand_dims(tgt, 0), [self.num_heads, 1])
            # Cast to same dtype for tf.stack compatibility
            head_indices = tf.cast(tf.repeat(tf.range(self.num_heads), num_edges), tf.int64)
            tgt_flat = tf.cast(tf.reshape(tgt_expanded, [-1]), tf.int64)
            indices = tf.stack(
                [head_indices, tgt_flat],
                axis=1,
            )
            updates = tf.reshape(weighted_messages, [-1, self.head_dim])
            output = tf.tensor_scatter_nd_add(output, indices, updates)
            return output

        def no_edges() -> Any:
            return h

        output = tf.cond(num_edges_tensor > 0, attention_pass, no_edges)

        # Combine heads
        if self.concat_heads:
            # (num_nodes, num_heads * head_dim)
            output = tf.transpose(output, [1, 0, 2])
            output = tf.reshape(output, (num_nodes, -1))
        else:
            # Average over heads
            output = tf.reduce_mean(output, axis=0)

        return self.activation(output)

    def compute_output_shape(self, input_shape: Any) -> tuple[Any, int]:
        """Compute output shape for Keras 3.x compatibility."""
        if self.concat_heads:
            return (input_shape[0][0], self.output_dim)
        else:
            return (input_shape[0][0], self.head_dim)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_heads": self.num_heads,
                "concat_heads": self.concat_heads,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation_name,
            }
        )
        result: dict[str, Any] = config
        return result


class VulnerabilityGNN:
    """GNN model for vulnerability prediction in SBOM graphs.

    Supports:
    - Node classification: Predict if each package is vulnerable
    - Graph classification: Predict overall supply chain risk
    """

    def __init__(self, config: GNNConfig | None = None):
        """Initialize the GNN model.

        Args:
            config: Model configuration
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for VulnerabilityGNN")

        self.config = config or GNNConfig()
        self.model: keras.Model | None = None
        self._build_model()

    def _build_model(self) -> None:
        """Build the GNN model architecture."""
        cfg = self.config

        # Inputs
        node_features = keras.Input(shape=(cfg.input_dim,), name="node_features")
        edge_index = keras.Input(shape=(2,), dtype=tf.int64, name="edge_index")
        num_nodes = keras.Input(shape=(), dtype=tf.int32, name="num_nodes")

        # Initial projection
        h = keras.layers.Dense(cfg.hidden_dim, activation="relu")(node_features)
        h = keras.layers.Dropout(cfg.dropout_rate)(h)

        # Transpose edge_index using custom layer for Keras 3.x compatibility
        edge_index_t = TransposeLayer(name="edge_transpose")(edge_index)

        # Graph convolution layers
        for i in range(cfg.num_layers):
            if cfg.use_attention and i < cfg.num_layers - 1:
                # Use GAT for intermediate layers
                layer = GraphAttentionLayer(
                    output_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    dropout_rate=cfg.dropout_rate,
                    name=f"gat_{i}",
                )
            else:
                # Use GCN
                layer = GraphConvLayer(
                    output_dim=cfg.hidden_dim,
                    normalize=cfg.normalize,
                    name=f"gcn_{i}",
                )

            h = layer([h, edge_index_t, num_nodes[0]])
            h = keras.layers.Dropout(cfg.dropout_rate)(h)

        # Node classification head
        node_logits = keras.layers.Dense(cfg.num_classes, name="node_classifier")(h)

        # Build model
        self.model = keras.Model(
            inputs=[node_features, edge_index, num_nodes],
            outputs=node_logits,
            name="VulnerabilityGNN",
        )

        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=cfg.learning_rate,
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(
        self,
        graphs: list[GraphData],
        epochs: int | None = None,
        validation_split: float = 0.1,
    ) -> dict[str, list[float]]:
        """Train the model on a list of graphs.

        Args:
            graphs: List of GraphData objects with labels
            epochs: Number of training epochs
            validation_split: Fraction for validation

        Returns:
            Training history
        """
        if not graphs:
            raise ValueError("No training graphs provided")

        epochs = epochs or self.config.epochs

        # Prepare training data
        all_features = []
        all_edges = []
        all_labels = []
        all_num_nodes = []

        for g in graphs:
            if g.node_labels is None:
                continue
            all_features.append(g.node_features)
            # Pad edge index to fixed size (will be variable in practice)
            all_edges.append(g.edge_index.T)  # (num_edges, 2)
            all_labels.append(g.node_labels)
            all_num_nodes.append(g.num_nodes)

        if not all_features:
            raise ValueError("No labeled graphs for training")

        # For simplicity, train on each graph separately
        # In production, use proper batching with padding
        history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(epochs):
            epoch_losses = []
            epoch_accs = []

            for features, edges, labels, num_nodes in zip(
                all_features, all_edges, all_labels, all_num_nodes, strict=False
            ):
                # Prepare inputs - GNN processes nodes as batch, no extra dimension
                x = {
                    "node_features": features,  # shape: (num_nodes, input_dim)
                    "edge_index": edges,  # shape: (num_edges, 2)
                    "num_nodes": np.array([num_nodes]),  # shape: (1,) scalar
                }
                y = labels  # shape: (num_nodes,)

                # Train step
                assert self.model is not None, "Model not initialized"
                result = self.model.train_on_batch(x, y)
                epoch_losses.append(result[0])
                epoch_accs.append(result[1])

            history["loss"].append(np.mean(epoch_losses))
            history["accuracy"].append(np.mean(epoch_accs))

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"loss={history['loss'][-1]:.4f}, "
                    f"acc={history['accuracy'][-1]:.4f}"
                )

        return history

    def predict(self, graph: GraphData) -> np.ndarray:
        """Predict vulnerability labels for a graph.

        Args:
            graph: GraphData to predict on

        Returns:
            Predicted labels for each node
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # GNN inputs have different batch sizes (nodes vs edges vs scalar)
        # Use direct model call instead of model.predict() for Keras 3.x compatibility
        # num_nodes must be 1D array [n] to match Input(shape=()) which expects (batch_size,)
        x = [
            tf.constant(graph.node_features, dtype=tf.float32),
            tf.constant(graph.edge_index.T, dtype=tf.int64),
            tf.constant([graph.num_nodes], dtype=tf.int32),  # 1D array [n]
        ]

        logits = self.model(x, training=False)
        predictions = np.argmax(logits.numpy(), axis=-1)
        return np.asarray(predictions)

    def predict_proba(self, graph: GraphData) -> np.ndarray:
        """Predict vulnerability probabilities for a graph.

        Args:
            graph: GraphData to predict on

        Returns:
            Probability distribution over classes for each node
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # GNN inputs have different batch sizes (nodes vs edges vs scalar)
        # Use direct model call instead of model.predict() for Keras 3.x compatibility
        # num_nodes must be 1D array [n] to match Input(shape=()) which expects (batch_size,)
        x = [
            tf.constant(graph.node_features, dtype=tf.float32),
            tf.constant(graph.edge_index.T, dtype=tf.int64),
            tf.constant([graph.num_nodes], dtype=tf.int32),  # 1D array [n]
        ]

        logits = self.model(x, training=False)
        probs = tf.nn.softmax(logits).numpy()
        return np.asarray(probs)

    def evaluate(self, graphs: list[GraphData]) -> dict[str, float]:
        """Evaluate model performance on test graphs.

        Args:
            graphs: List of labeled GraphData objects

        Returns:
            Evaluation metrics
        """
        predictions_list: list[Any] = []
        labels_list: list[Any] = []

        for g in graphs:
            if g.node_labels is None:
                continue

            preds = self.predict(g)
            predictions_list.extend(preds)
            labels_list.extend(g.node_labels)

        if not labels_list:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        all_predictions = np.array(predictions_list)
        all_labels = np.array(labels_list)

        # Accuracy
        accuracy = float(np.mean(all_predictions == all_labels))

        # Per-class metrics
        metrics: dict[str, float] = {"accuracy": accuracy}

        for cls in range(self.config.num_classes):
            cls_mask = all_labels == cls
            pred_mask = all_predictions == cls

            tp = np.sum(cls_mask & pred_mask)
            fp = np.sum(~cls_mask & pred_mask)
            fn = np.sum(cls_mask & ~pred_mask)

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (
                float(2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )

            metrics[f"class_{cls}_precision"] = precision
            metrics[f"class_{cls}_recall"] = recall
            metrics[f"class_{cls}_f1"] = f1

        return metrics

    def save(self, path: str) -> None:
        """Save model weights to file."""
        if self.model is not None:
            self.model.save_weights(path)
            logger.info(f"Saved model to {path}")

    def load(self, path: str) -> None:
        """Load model weights from file."""
        if self.model is not None:
            self.model.load_weights(path)
            logger.info(f"Loaded model from {path}")


class SimpleVulnerabilityClassifier:
    """Simple non-GNN classifier for comparison and when TensorFlow unavailable.

    Uses package features directly without message passing.
    """

    def __init__(self, num_classes: int = 3):
        """Initialize classifier."""
        self.num_classes = num_classes
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, graphs: list[GraphData], epochs: int = 100) -> None:
        """Train the simple classifier using gradient descent."""
        # Collect all features and labels
        all_features = []
        all_labels = []

        for g in graphs:
            if g.node_labels is not None:
                all_features.append(g.node_features)
                all_labels.append(g.node_labels)

        if not all_features:
            raise ValueError("No labeled data")

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        # Initialize weights
        input_dim = X.shape[1]
        self.weights = np.random.randn(input_dim, self.num_classes) * 0.01
        self.bias = np.zeros(self.num_classes)

        # Train with gradient descent
        lr = 0.01
        for epoch in range(epochs):
            # Forward pass
            logits = X @ self.weights + self.bias
            probs = self._softmax(logits)

            # Loss
            loss = self._cross_entropy(probs, y)

            # Backward pass
            n_samples = X.shape[0]
            grad_logits = probs.copy()
            grad_logits[np.arange(n_samples), y] -= 1
            grad_logits /= n_samples

            grad_weights = X.T @ grad_logits
            grad_bias = np.mean(grad_logits, axis=0)

            # Update
            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias

            if (epoch + 1) % 20 == 0:
                acc = np.mean(np.argmax(probs, axis=1) == y)
                logger.info(f"Epoch {epoch + 1}: loss={loss:.4f}, acc={acc:.4f}")

    def predict(self, graph: GraphData) -> np.ndarray:
        """Predict labels for graph nodes."""
        if self.weights is None:
            raise RuntimeError("Model not trained")

        logits = graph.node_features @ self.weights + self.bias
        return np.asarray(np.argmax(logits, axis=1))

    def predict_proba(self, graph: GraphData) -> np.ndarray:
        """Predict class probabilities."""
        if self.weights is None:
            raise RuntimeError("Model not trained")

        logits = graph.node_features @ self.weights + self.bias
        return self._softmax(logits)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        result: np.ndarray = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return result

    def _cross_entropy(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        n = len(labels)
        return float(-np.mean(np.log(probs[np.arange(n), labels] + 1e-8)))


if __name__ == "__main__":
    # Test the GNN model
    from medtech_ai_security.sbom_analysis.graph_builder import SBOMGraphBuilder
    from medtech_ai_security.sbom_analysis.parser import SBOMParser, create_sample_sbom

    print("[+] Testing VulnerabilityGNN")

    # Create sample data
    sample_sbom = create_sample_sbom()
    parser = SBOMParser()
    dep_graph = parser.parse_json(sample_sbom)

    builder = SBOMGraphBuilder()
    graph_data = builder.build(dep_graph)

    print(f"    Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    print(f"    Labels: {graph_data.node_labels}")

    if TF_AVAILABLE:
        # Test GNN
        config = GNNConfig(
            input_dim=graph_data.node_features.shape[1],
            hidden_dim=32,
            num_layers=2,
            epochs=50,
        )
        model = VulnerabilityGNN(config)

        # Train
        print("\n[+] Training GNN...")
        history = model.train([graph_data], epochs=50)
        print(f"    Final loss: {history['loss'][-1]:.4f}")
        print(f"    Final accuracy: {history['accuracy'][-1]:.4f}")

        # Predict
        predictions = model.predict(graph_data)
        print(f"\n[+] Predictions: {predictions}")
        print(f"    True labels: {graph_data.node_labels}")

        # Evaluate
        metrics = model.evaluate([graph_data])
        print(f"\n[+] Metrics: {metrics}")
    else:
        print("[!] TensorFlow not available, testing simple classifier")
        classifier = SimpleVulnerabilityClassifier()
        classifier.fit([graph_data], epochs=100)

        predictions = classifier.predict(graph_data)
        print(f"\n[+] Predictions: {predictions}")
        print(f"    True labels: {graph_data.node_labels}")

    print("\n[OK] GNN model test complete!")
