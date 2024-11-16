
import pandas as pd
import plotly.graph_objects as go

file_path = '...scores.csv'
df = pd.read_csv(file_path)

pc1 = 'PC 1'
pc2 = 'PC 2'
pc3 = 'PC 3'

color_scale = df[pc1] + df[pc2] + df[pc3]

fig = go.Figure(data=[go.Scatter3d(
    x=df[pc1],
    y=df[pc2],
    z=df[pc3],
    mode='markers',
    marker=dict(
        size=5,
        color=color_scale,
        colorscale='Magma',
        colorbar=dict(title='Color Scale'),
        opacity=0.8
    )
)])
fig.update_layout(
    scene=dict(
        xaxis_title=pc1,
        yaxis_title=pc2,
        zaxis_title=pc3,
    ),
    title='Scores'
)

fig.write_html('3d_plot_with_color_scale.html', auto_open=True)
