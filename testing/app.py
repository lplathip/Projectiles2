from flask import Flask, render_template, jsonify, request
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

def compute_bouncing_trajectory(u, theta_deg, h, COR, g=9.81, dt=0.01, N=10):
    theta = np.deg2rad(theta_deg)
    vx = u * np.cos(theta)
    vy = u * np.sin(theta)
    
    x_data = [0]
    y_data = [h]
    t_data = [0]

    x = 0
    y = h
    t = 0
    bounce_count = 0

    while bounce_count < N and y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        ax = 0
        ay = -g
        
        x_new = x + vx * dt + 0.5 * ax * dt**2
        y_new = y + vy * dt + 0.5 * ay * dt**2
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        
        if y_new < 0:
            y_new = 0
            vy_new = -COR * vy_new
            bounce_count += 1
        
        x = x_new
        y = y_new
        vx = vx_new
        vy = vy_new
        t += dt

        x_data.append(x)
        y_data.append(y)
        t_data.append(t)
    
    return x_data, y_data

def create_bouncing_trajectory_figure(x_data, y_data, dt=0.01):
    N = len(x_data)
    frames = [go.Frame(
        data=[go.Scatter(
            x=[x_data[k]],
            y=[y_data[k]],
            mode="markers",
            marker=dict(color="red", size=10))],
        name=str(k)
    ) for k in range(N)]

    fig = go.Figure(
        data=[go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
            line=dict(width=2, color="blue"))],
        layout=go.Layout(
            xaxis=dict(range=[min(x_data) - 1, max(x_data) + 1], autorange=False, zeroline=False),
            yaxis=dict(range=[min(y_data) - 1, max(y_data) + 1], autorange=False, zeroline=False),
            title_text="Projectile Motion with Bouncing",
            hovermode="closest",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 10, "redraw": True}, "fromcurrent": True}])
                ])]
        ),
        frames=frames
    )

    return fig

@app.route('/page8')
def index_8():
    return render_template('task8.html')

@app.route('/update_8', methods=['POST'])
def update_8():
    data = request.json
    u = float(data['speed'])
    theta_deg = float(data['angle'])
    h = float(data['height'])
    COR = float(data['cor'])

    x_data, y_data = compute_bouncing_trajectory(u, theta_deg, h, COR)
    fig = create_bouncing_trajectory_figure(x_data, y_data)
    graphJSON = pio.to_json(fig)
    return jsonify(graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
