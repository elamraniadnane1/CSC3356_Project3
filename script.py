from flask import Flask, render_template
import plotly.graph_objects as go
import networkx as nx
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data from files
node_labels_df = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\soc-political-retweet\\soc-political-retweet_node_labels.csv', header=None, names=['node', 'label'])
edges_df = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\soc-political-retweet\\soc-political-retweet_edges.csv', header=None, names=['source', 'target', 'timestamp'])

# Convert DataFrame to dictionary for labels
node_labels = pd.Series(node_labels_df.label.values,index=node_labels_df.node).to_dict()

# Create the graph
G = nx.Graph()
for node, label in node_labels.items():
    G.add_node(node, label=label)

# Adding edges from DataFrame
G.add_edges_from(edges_df[['source', 'target']].values)

# Color mapping based on political alignment
color_map = {1: 'red', 2: 'blue'}  # 1: right, 2: left
node_colors = [color_map[G.nodes[node]['label']] for node in G if 'label' in G.nodes[node]]


app = Flask(__name__)

@app.route('/')
def index():
    # Assuming 'G' has been previously defined and populated with nodes and edges
    pos = nx.spring_layout(G)  # Position nodes using a layout algorithm

    # Prepare data for Plotly plotting
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Node: {node}<br>Label: {G.nodes[node]["label"]}')
        node_colors.append(color_map[G.nodes[node]['label']])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            color=node_colors,
            colorbar=dict(
                title='Political Alignment',
                tickvals=[1, 2],
                ticktext=['Right', 'Left']
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Visualization of Social-Political Retweet Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

    # Convert the figure to HTML and return it
    graphHTML = fig.to_html(full_html=False)
    return render_template('C:\\Users\\DELL\\OneDrive\\Desktop\\CSC3356_Project3\\templates\\index.html', graphHTML=graphHTML)

if __name__ == '__main__':
    app.run(debug=True)
