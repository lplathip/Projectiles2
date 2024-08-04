import plotly.graph_objs as go
import plotly.io as pio

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 11, 12, 13, 14]

# Create a trace
trace = go.Scatter(
    x=x,
    y=y,
    mode='lines+markers',
    name='Sample Data'
)

# Create layout with black background
layout = go.Layout(
    title='Sample Plot',
    xaxis=dict(title='X-axis'),
    yaxis=dict(title='Y-axis'),
    paper_bgcolor='black',
    plot_bgcolor='black',
    font=dict(color='white')  # Change font color to white for better visibility
)

# Create figure with the trace and layout
fig = go.Figure(data=[trace], layout=layout)

# Display the figure
pio.show(fig)
