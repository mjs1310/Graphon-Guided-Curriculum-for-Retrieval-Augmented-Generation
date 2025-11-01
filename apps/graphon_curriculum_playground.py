"""Streamlit playground to explore graphon-guided RAG curricula."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass(frozen=True)
class KernelOption:
    """Description of an available graphon kernel."""

    name: str
    description: str

    def evaluate(self, u: np.ndarray, v: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Return the probability matrix for the kernel."""

        if self.name == "Block Party":
            high = params.get("in_block", 0.7)
            low = params.get("out_block", 0.15)
            same_block = ((u < 0.5) & (v < 0.5)) | ((u >= 0.5) & (v >= 0.5))
            return np.where(same_block, high, low)

        if self.name == "Ripple":
            freq = params.get("frequency", 3.0)
            offset = params.get("offset", 0.15)
            amplitude = params.get("amplitude", 0.7)
            wave = 0.5 * (np.sin(freq * math.pi * (u + v)) + 1.0)
            return np.clip(offset + amplitude * wave, 0.0, 1.0)

        if self.name == "Spotlight":
            center = params.get("center", 0.5)
            sharpness = params.get("sharpness", 25.0)
            baseline = params.get("baseline", 0.1)
            gaussian = np.exp(-sharpness * ((u - center) ** 2 + (v - center) ** 2))
            return np.clip(baseline + (1 - baseline) * gaussian, 0.0, 1.0)

        raise ValueError(f"Unknown kernel: {self.name}")


def load_default_curriculum() -> Dict[str, float]:
    """Load reference parameters from the project default configuration."""

    if not DEFAULT_CONFIG_PATH.exists():
        return {
            "ramp_steps": 8,
            "replay_fraction": 0.2,
            "max_batch_size": 32,
        }

    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    curriculum = config.get("curriculum", {})
    return {
        "ramp_steps": curriculum.get("warmup_steps", 8),
        "replay_fraction": curriculum.get("replay_fraction", 0.2),
        "max_batch_size": curriculum.get("max_batch_size", 32),
    }


def generate_latents(num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.random(num_nodes))


def build_probability_matrix(
    latents: np.ndarray, kernel: KernelOption, params: Dict[str, float]
) -> np.ndarray:
    u = latents[:, None]
    v = latents[None, :]
    return kernel.evaluate(u, v, params)


def sample_graph(probabilities: np.ndarray, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.random(probabilities.shape), k=1)
    adjacency = (upper < np.triu(probabilities, k=1)).astype(int)
    adjacency = adjacency + adjacency.T
    graph = nx.from_numpy_array(adjacency)
    return graph


def _normalize(values: Dict[int, float]) -> Dict[int, float]:
    arr = np.array(list(values.values()), dtype=float)
    if arr.sum() == 0:
        return {node: 0.0 for node in values}
    arr = arr / arr.sum()
    return {node: val for node, val in zip(values.keys(), arr)}


def compute_difficulty_scores(graph: nx.Graph) -> pd.DataFrame:
    """Combine simple heuristics into a playful difficulty score."""

    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    pagerank = nx.pagerank(graph, alpha=0.9) if graph.number_of_edges() else {n: 1.0 for n in graph.nodes()}

    degree_norm = _normalize(degree)
    clustering_norm = _normalize(clustering)
    pagerank_norm = _normalize(pagerank)

    data = []
    for node in graph.nodes():
        score = 0.5 * degree_norm[node] + 0.3 * clustering_norm[node] + 0.2 * pagerank_norm[node]
        data.append({
            "node": node,
            "degree": degree[node],
            "clustering": clustering[node],
            "pagerank": pagerank[node],
            "difficulty": score,
        })

    frame = pd.DataFrame(data)
    return frame.sort_values("difficulty", ascending=False)


def build_curriculum_plan(
    num_items: int,
    steps: int,
    ramp_steps: int,
    replay_fraction: float,
    max_batch_size: int,
) -> pd.DataFrame:
    ramp_steps = max(1, min(ramp_steps, steps))
    replay_fraction = np.clip(replay_fraction, 0.0, 0.9)

    primary_schedule = np.linspace(0.2, 1.0, ramp_steps)
    if steps > ramp_steps:
        tail = np.full(steps - ramp_steps, 1.0)
        primary_schedule = np.concatenate([primary_schedule, tail])

    primary_counts = np.ceil(primary_schedule * max_batch_size).astype(int)
    replay_counts = np.ceil(primary_counts * replay_fraction).astype(int)

    rows = []
    for step in range(steps):
        rows.append({
            "step": step + 1,
            "new_items": int(primary_counts[step]),
            "replay_items": int(replay_counts[step]),
            "cumulative_new": int(primary_counts[: step + 1].sum()),
        })

    frame = pd.DataFrame(rows)
    frame["cumulative_new"] = frame["cumulative_new"].clip(upper=num_items)
    return frame


def make_graph_figure(graph: nx.Graph) -> go.Figure:
    if graph.number_of_nodes() == 0:
        return go.Figure()

    layout = nx.spring_layout(graph, seed=42, dim=2)
    x_nodes = [layout[node][0] for node in graph.nodes()]
    y_nodes = [layout[node][1] for node in graph.nodes()]

    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        edge_x.extend([layout[u][0], layout[v][0], None])
        edge_y.extend([layout[u][1], layout[v][1], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#AAAAAA"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers",
        marker=dict(
            size=[5 + 4 * graph.degree(node) for node in graph.nodes()],
            color=list(nx.clustering(graph).values()),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Clustering"),
        ),
        text=[f"Node {node}" for node in graph.nodes()],
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    return fig


KERNELS: Tuple[KernelOption, ...] = (
    KernelOption(
        name="Block Party",
        description="Two communities with strong intra-links. Great for demonstrating curriculum replay.",
    ),
    KernelOption(
        name="Ripple",
        description="A wavy latent structure. Highlights boundary examples that are hard to classify.",
    ),
    KernelOption(
        name="Spotlight",
        description="All attention on a latent hotspot. Perfect for dense retrieval neighborhoods.",
    ),
)


st.set_page_config(page_title="Graphon Curriculum Playground", layout="wide")
st.title("Graphon Curriculum Playground")
st.caption(
    "Experiment with synthetic corpora, difficulty signals, and curriculum schedules inspired by the project pipeline."
)

with st.sidebar:
    st.header("Graphon Builder")
    num_nodes = st.slider("Number of passages", min_value=12, max_value=120, value=48, step=6)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=7)

    kernel_names = [kernel.name for kernel in KERNELS]
    kernel_index = st.selectbox("Latent kernel", range(len(KERNELS)), format_func=lambda idx: kernel_names[idx])
    kernel = KERNELS[kernel_index]

    st.markdown(f"**{kernel.name}:** {kernel.description}")

    kernel_params: Dict[str, float] = {}
    if kernel.name == "Block Party":
        kernel_params["in_block"] = st.slider("In-block connectivity", 0.1, 1.0, 0.72, 0.02)
        kernel_params["out_block"] = st.slider("Cross-block connectivity", 0.0, 0.6, 0.18, 0.02)
    elif kernel.name == "Ripple":
        kernel_params["frequency"] = st.slider("Frequency", 1.0, 6.0, 3.0, 0.5)
        kernel_params["offset"] = st.slider("Offset", 0.0, 0.6, 0.15, 0.01)
        kernel_params["amplitude"] = st.slider("Amplitude", 0.2, 1.0, 0.65, 0.05)
    elif kernel.name == "Spotlight":
        kernel_params["center"] = st.slider("Hotspot center", 0.0, 1.0, 0.5, 0.05)
        kernel_params["sharpness"] = st.slider("Sharpness", 5.0, 60.0, 25.0, 1.0)
        kernel_params["baseline"] = st.slider("Baseline", 0.0, 0.5, 0.12, 0.01)

    st.header("Curriculum")
    defaults = load_default_curriculum()
    curriculum_steps = st.slider("Total training steps", 4, 24, 12, 1)
    ramp_steps = st.slider("Warmup steps", 1, curriculum_steps, int(defaults["ramp_steps"]), 1)
    replay_fraction = st.slider("Replay fraction", 0.0, 0.8, float(defaults["replay_fraction"]), 0.05)
    max_batch_size = st.slider("Max passages per batch", 8, 128, int(defaults["max_batch_size"]), 8)

latents = generate_latents(num_nodes, seed)
probabilities = build_probability_matrix(latents, kernel, kernel_params)
graph = sample_graph(probabilities, seed)

graph_col, stats_col = st.columns([3, 2])

with graph_col:
    st.subheader("Synthetic corpus graph")
    figure = make_graph_figure(graph)
    st.plotly_chart(figure, use_container_width=True)

with stats_col:
    st.subheader("Graph summary")
    st.metric("Nodes", graph.number_of_nodes())
    st.metric("Edges", graph.number_of_edges())
    st.metric("Avg. clustering", f"{nx.average_clustering(graph):.2f}" if graph.number_of_edges() else "0.00")
    st.metric("Largest component", len(max(nx.connected_components(graph), key=len)) if graph.number_of_nodes() else 0)

st.markdown("---")

scores = compute_difficulty_scores(graph)
top_k = min(10, len(scores))

st.subheader("Difficulty sampler")
st.write(
    "We blend degree, clustering, and PageRank to create a playful difficulty signal. "
    "Use it to cherry-pick tricky passages for evaluation or replay."
)
st.dataframe(scores.head(top_k).style.format({"clustering": "{:.2f}", "pagerank": "{:.3f}", "difficulty": "{:.3f}"}))

curriculum = build_curriculum_plan(num_nodes, curriculum_steps, ramp_steps, replay_fraction, max_batch_size)

st.subheader("Curriculum schedule")
st.write(
    "The schedule ramps up new passages while mixing in replayed content—just like the Graphon-guided curriculum."
)
st.dataframe(curriculum)

st.success(
    "Tip: Swap kernels or seeds to see how curriculum demands change. "
    "This playground doubles as a quick sanity-check for graph-driven retrieval experiments."
)
