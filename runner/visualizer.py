import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


NODE_COLORS = {
    "acquirer": "#4C8EDA",
    "holding": "#F5A623",
    "propco": "#5DB85C",
}


def visualize_graph(graph: dict, title: str, output_path: str):
    """
    Render an ownership graph as a directed network diagram.

    Nodes are colored by type (acquirer, holding, propco) and edges are
    labelled with ownership percentages.

    Args:
        graph: Dict with 'nodes' and 'edges' keys following the test case schema.
        title: Title displayed above the graph.
        output_path: File path to save the figure.
    """
    G = nx.DiGraph()

    color_map = []
    for node in graph["nodes"]:
        G.add_node(node["id"], label=node["name"])
        color_map.append(NODE_COLORS.get(node["type"], "#cccccc"))

    edge_labels = {}
    for edge in graph["edges"]:
        G.add_edge(edge["from"], edge["to"])
        edge_labels[(edge["from"], edge["to"])] = f"{edge['percentage']}%"

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        labels=nx.get_node_attributes(G, "label"),
        node_color=color_map,
        node_size=2000,
        font_size=9,
        arrows=True,
        arrowsize=20,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graph saved to {output_path}")
