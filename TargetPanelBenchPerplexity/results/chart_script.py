import plotly.graph_objects as go
import pandas as pd

# Data from the provided JSON
data = {
    "methods": ["Archipelago AEA", "CMA-ES Standard", "PSO Adaptive", "Simple Score Rank"],
    "precision_at_20": [0.55, 0.42, 0.38, 0.25],
    "panel_recall": [0.85, 0.75, 0.70, 0.60],
    "network_diversity": [7.23, 4.20, 3.80, 2.10],
    "overall_score": [0.712, 0.580, 0.520, 0.350]
}

# Normalize network diversity to 0-1 scale to match other metrics
max_diversity = max(data["network_diversity"])
normalized_diversity = [x / max_diversity for x in data["network_diversity"]]

# Define colors from the brand palette
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F']

# Create the grouped bar chart
fig = go.Figure()

# Add bars for each metric
fig.add_trace(go.Bar(
    name='Precision@20',
    x=data["methods"],
    y=data["precision_at_20"],
    marker_color=colors[0],
    cliponaxis=False,
    hovertemplate='<b>%{x}</b><br>Precision@20: %{y:.2f}<extra></extra>'
))

fig.add_trace(go.Bar(
    name='Panel Recall',
    x=data["methods"],
    y=data["panel_recall"],
    marker_color=colors[1],
    cliponaxis=False,
    hovertemplate='<b>%{x}</b><br>Panel Recall: %{y:.2f}<extra></extra>'
))

fig.add_trace(go.Bar(
    name='Net Diversity',
    x=data["methods"],
    y=normalized_diversity,
    marker_color=colors[2],
    cliponaxis=False,
    customdata=data["network_diversity"],
    hovertemplate='<b>%{x}</b><br>Net Diversity: %{customdata:.2f}<br>(Normalized: %{y:.2f})<extra></extra>'
))

fig.add_trace(go.Bar(
    name='Overall Score',
    x=data["methods"],
    y=data["overall_score"],
    marker_color=colors[3],
    cliponaxis=False,
    hovertemplate='<b>%{x}</b><br>Overall Score: %{y:.2f}<extra></extra>'
))

# Update layout for professional appearance
fig.update_layout(
    title='TargetPanelBench Performance Comparison',
    xaxis_title='Methods',
    yaxis_title='Normalized Score',
    barmode='group',
    legend=dict(
        orientation='h', 
        yanchor='bottom', 
        y=1.05, 
        xanchor='center', 
        x=0.5
    )
)

# Update axes
fig.update_yaxes(range=[0, 1])

# Save the chart
fig.write_image('targetpanel_performance_comparison.png')
print("Chart saved as 'targetpanel_performance_comparison.png'")