import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


NODE_COLORS = {
    "acquirer": "#4C8EDA",
    "holding": "#F5A623",
    "propco": "#5DB85C",
}

# Palette for acquirer group coloring — each group gets a distinct color
GROUP_COLORS = [
    "#E74C3C",  # red
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#E67E22",  # dark orange
    "#2ECC71",  # emerald
    "#3498DB",  # bright blue
    "#F39C12",  # yellow-orange
    "#E91E63",  # pink
]


def visualize_graph(graph: dict, title: str, output_path: str,
                    acquirer_groups: list = None):
    """
    Render an ownership graph as a directed network diagram.

    Nodes are colored by type (acquirer, holding, propco). When acquirer
    groups are provided, members of the same group share a distinct color.

    Args:
        graph: Dict with 'nodes' and 'edges' keys following the test case schema.
        title: Title displayed above the graph.
        output_path: File path to save the figure.
        acquirer_groups: Optional list of group dicts with 'id' and 'members'.
    """
    G = nx.DiGraph()

    # Build a mapping from acquirer ID -> group color
    member_color = {}
    for idx, group in enumerate(acquirer_groups or []):
        color = GROUP_COLORS[idx % len(GROUP_COLORS)]
        for member_id in group["members"]:
            member_color[member_id] = color

    color_map = []
    for node in graph["nodes"]:
        G.add_node(node["id"], label=node["name"])
        if node["id"] in member_color:
            color_map.append(member_color[node["id"]])
        else:
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

    # Add legend for groups
    if acquirer_groups:
        from matplotlib.patches import Patch
        legend_elements = []
        for idx, group in enumerate(acquirer_groups):
            color = GROUP_COLORS[idx % len(GROUP_COLORS)]
            legend_elements.append(Patch(facecolor=color, label=f"Group {group['id']}"))
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graph saved to {output_path}")
