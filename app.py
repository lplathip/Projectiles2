from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


# Task 1: Simple Projectile Motion
def compute_trajectory_1(u, theta_deg, g, h):
    theta = np.deg2rad(theta_deg)
    dt = 0.01

    vx = u * np.cos(theta)
    vy = u * np.sin(theta)

    x_data = [0]
    y_data = [h]

    x = 0
    y = h
    t = 0
    while y >= 0:
        x += vx * dt
        y += vy * dt - 0.5 * g * dt**2
        vy -= g * dt
        t += dt
        x_data.append(x)
        y_data.append(y)
    
    return x_data, y_data

@app.route('/page1')
def index_1():
    return render_template('task1.html')

@app.route('/update_1', methods=['POST'])
def update_1():
    g = float(request.json['gravity'])
    theta_deg = float(request.json['angle'])
    u = float(request.json['speed'])
    h = float(request.json['height'])
    
    x_values, y_values = compute_trajectory_1(u, theta_deg, g, h)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Trajectory'))
    fig.update_layout(title='Simple Projectile Motion Model',
                      xaxis_title='x (m)',
                      yaxis_title='y (m)',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'))

    graphJSON = pio.to_json(fig)
    return jsonify(graphJSON)

# Task 2: Exact Projectile Motion
def compute_trajectory_2(g, theta_deg, u, h):
    theta = np.deg2rad(theta_deg)
    
    T = (u * np.sin(theta) + np.sqrt((u * np.sin(theta))**2 + 2 * g * h)) / g
    R = u * np.cos(theta) * T
    xa = (u**2 * np.sin(2 * theta)) / (2 * g)
    ya = h + (u**2 * np.sin(theta)**2) / (2 * g)
    
    x_values = np.linspace(0, R, num=500)
    y_values = h + x_values * np.tan(theta) - (g * (1 + np.tan(theta)**2) * x_values**2) / (2 * u**2)
    
    valid_indices = y_values >= 0
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]
    
    return x_values, y_values, xa, ya, T

@app.route('/page2')
def index_2():
    return render_template('task2.html')

@app.route('/update_2', methods=['POST'])
def update_2():
    g = float(request.json['gravity'])
    theta_deg = float(request.json['angle'])
    u = float(request.json['speed'])
    h = float(request.json['height'])
    
    x_values, y_values, xa, ya, T = compute_trajectory_2(g, theta_deg, u, h)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Trajectory'))
    fig.add_trace(go.Scatter(x=[xa], y=[ya], mode='markers', name='Apogee', marker=dict(color='red', size=10)))
    fig.update_layout(title='Exact Projectile Motion Model',
                      xaxis_title='x (m)',
                      yaxis_title='y (m)',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'))
    graphJSON = pio.to_json(fig)
    return jsonify(graphJSON=graphJSON, time_of_flight=T)

# Task 3: Launch Angles and Trajectories

def calculateMinimumLaunchAngleAndSpeed_3(X=1000, Y=300, h=0, g=9.81):
    term1 = Y - h
    term2 = np.sqrt(X**2 + (Y - h)**2)
    min_u_squared = g * (term1 + term2)
    
    if min_u_squared <= 0:
        raise ValueError("No valid solution for minimum speed.")
    
    min_u = np.sqrt(min_u_squared)
    
    theta = np.arctan((term1 + term2) / X)
    theta_deg = np.degrees(theta)
    return min_u, theta_deg

def calculate_launch_angles_3(X=1000, Y=300, h=0, g=9.81, u=150):
    a = g * X**2 / (2 * u**2)
    b = -X
    c = Y - h + g * X**2 / (2 * u**2)
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None, None
    discriminant_sqrt = np.sqrt(discriminant)
    angle1_rad = np.arctan((-b + discriminant_sqrt) / (2 * a))
    angle2_rad = np.arctan((-b - discriminant_sqrt) / (2 * a))
    angle1_deg = np.degrees(angle1_rad)
    angle2_deg = np.degrees(angle2_rad)
    low_ball_angle = min(angle1_deg, angle2_deg)
    high_ball_angle = max(angle1_deg, angle2_deg)
    return low_ball_angle, high_ball_angle

def get_trajectory_data_3(angle, u, h, g):
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Coefficients for the quadratic equation
    a = 0.5 * g
    b = -u * np.sin(angle_rad)
    c = -h
    
    # Calculate discriminant
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return np.array([]), np.array([])  # No valid trajectory if discriminant is negative
    
    # Time of flight (we only consider the positive root)
    t_flight = (-b + np.sqrt(discriminant)) / (2 * a)
    
    # Generate time values and calculate corresponding x and y values
    t_max = np.linspace(0, t_flight, num=100)
    x_vals = u * np.cos(angle_rad) * t_max
    y_vals = h + u * np.sin(angle_rad) * t_max - 0.5 * g * t_max**2
    
    # Filter out y values below 0
    mask = y_vals >= 0
    return x_vals[mask], y_vals[mask]

def plot_projectile_trajectories_3(X=1000, Y=300, h=0, g=9.81, u=150):
    low_ball_angle, high_ball_angle = calculate_launch_angles_3(X, Y, h, g, u)

    fig = go.Figure()

    if low_ball_angle is not None and high_ball_angle is not None:
        # High Ball Trajectory
        x_high, y_high = get_trajectory_data_3(high_ball_angle, u, h, g)
        fig.add_trace(go.Scatter(x=x_high, y=y_high, mode='lines', name=f'High Ball Trajectory ({high_ball_angle:.1f}°)'))

        # Low Ball Trajectory
        x_low, y_low = get_trajectory_data_3(low_ball_angle, u, h, g)
        fig.add_trace(go.Scatter(x=x_low, y=y_low, mode='lines', name=f'Low Ball Trajectory ({low_ball_angle:.1f}°)'))

    # Minimum Launch Trajectory
    min_u, theta_deg = calculateMinimumLaunchAngleAndSpeed_3(X, Y, h, g)
    x_min, y_min = get_trajectory_data_3(theta_deg, min_u, h, g)
    fig.add_trace(go.Scatter(x=x_min, y=y_min, mode='lines', name=f'Minimum Launch Trajectory ({theta_deg:.1f}°)'))

    # Bounding Parabola
    x_bound = np.linspace(0, 10 * X, num=100)
    y_bound = h + (u**2) / (2 * g) - (g / (2 * u**2)) * x_bound**2
    mask_bound = y_bound >= 0  # Filter out y values below 0
    fig.add_trace(go.Scatter(x=x_bound[mask_bound], y=y_bound[mask_bound], mode='lines', name='Bounding Parabola'))

    # Maximum Horizontal Distance Trajectory
    max_horizontal_angle = np.pi / 4  # 45 degrees for maximum range
    max_horizontal_angle_deg = np.degrees(max_horizontal_angle)
    x_max, y_max = get_trajectory_data_3(max_horizontal_angle_deg, u, h, g)
    fig.add_trace(go.Scatter(x=x_max, y=y_max, mode='lines', name=f'Maximum Horizontal Path ({max_horizontal_angle_deg:.1f}°)'))

    fig.add_trace(go.Scatter(x=[X], y=[Y], mode='markers', marker=dict(color='white', size=10), name=f'Target Point ({X}, {Y})'))

    fig.update_layout(
        xaxis_title='Horizontal Distance (m)',
        yaxis_title='Vertical Distance (m)',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode='closest',
        showlegend=True
    )

    graphJSON = fig.to_json()
    return graphJSON

@app.route('/page3', methods=['GET'])
def index_3():
    plot = plot_projectile_trajectories_3()
    return render_template('task3.html', plot=plot, X=1000, Y=300, h=0, g=9.81, u=150)

@app.route('/update_3', methods=['POST'])
def update_3():
    data = request.json
    X = float(data['X'])
    Y = float(data['Y'])
    h = float(data['h'])
    g = float(data['g'])
    u = float(data['u'])
    plot = plot_projectile_trajectories_3(X, Y, h, g, u)
    return jsonify({'plot': plot})

# Task 4: Arc Length and Time of Flight
def calculateArcLengthOfProjectile(u, theta, g, h):
    theta_rad = np.radians(theta)
    X = (u**2 / g) * (np.sin(theta_rad) * np.cos(theta_rad) + np.cos(theta_rad) * np.sqrt(np.sin(theta_rad)**2 + 2 * g * h / u**2))
    tan_theta = np.tan(theta_rad)
    lower_limit = tan_theta - (g * X / u**2) * (1 + tan_theta**2)
    upper_limit = tan_theta

    def integral_part(z):
        return 0.5 * np.log(np.abs(np.sqrt(1 + z**2) + z)) + 0.5 * z * np.sqrt(1 + z**2)

    arc_length = ((u**2) / (g * (1 + tan_theta**2))) * (integral_part(upper_limit) - integral_part(lower_limit))
    return arc_length

def compute_trajectory_4(g, theta_deg, u, h):
    theta = np.deg2rad(theta_deg)
    
    T = (u * np.sin(theta) + np.sqrt((u * np.sin(theta))**2 + 2 * g * h)) / g
    R = u * np.cos(theta) * T
    xa = (u**2 * np.sin(2 * theta)) / (2 * g)
    ya = h + (u**2 * np.sin(theta)**2) / (2 * g)
    
    x_values = np.linspace(0, R, num=500)
    y_values = h + x_values * np.tan(theta) - (g * (1 + np.tan(theta)**2) * x_values**2) / (2 * u**2)
    
    valid_indices = y_values >= 0
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]

    theta_max = np.arcsin(1 / np.sqrt(2 + 2 * g * h / u**2))
    R_max = (u**2 / g) * (np.sin(2 * theta_max) + np.cos(theta_max) * np.sqrt(np.sin(theta_max)**2 + 2 * g * h / u**2))
    x_values_max = np.linspace(0, R_max, num=500)
    y_values_max = h + x_values_max * np.tan(theta_max) - (g * (1 + np.tan(theta_max)**2) * x_values_max**2) / (2 * u**2)

    valid_indices_max = y_values_max >= 0
    x_values_max = x_values_max[valid_indices_max]
    y_values_max = y_values_max[valid_indices_max]
    
    T_max = (u * np.sin(theta_max) + np.sqrt((u * np.sin(theta_max))**2 + 2 * g * h)) / g
    
    arc_length = calculateArcLengthOfProjectile(u, theta_deg, g, h)
    arc_length_max = calculateArcLengthOfProjectile(u, np.degrees(theta_max), g, h)
    
    return (x_values, y_values, xa, ya, T, T_max, arc_length, arc_length_max, x_values_max, y_values_max)

@app.route('/page4')
def index_4():
    return render_template('task4.html')

@app.route('/update_4', methods=['POST'])
def update_4():
    g = float(request.json['gravity'])
    theta_deg = float(request.json['angle'])
    u = float(request.json['speed'])
    h = float(request.json['height'])
    
    x_values, y_values, xa, ya, T, T_max, arc_length, arc_length_max, x_values_max, y_values_max = compute_trajectory_4(g, theta_deg, u, h)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='User-defined Trajectory'))
    fig.add_trace(go.Scatter(x=x_values_max, y=y_values_max, mode='lines', name='Maximum Range Trajectory', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[xa], y=[ya], mode='markers', name='Apogee', marker=dict(color='red', size=10)))
    fig.update_layout(title='Exact Projectile Motion Model',
                      xaxis_title='x (m)',
                      yaxis_title='y (m)',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'))
    graphJSON = pio.to_json(fig)
    return jsonify(
        graphJSON=graphJSON,
        time_of_flight=round(T, 3),
        max_time_of_flight=round(T_max, 3),
        arc_length=round(arc_length, 3),
        max_arc_length=round(arc_length_max, 3)
    )

#Task7

u = 10  # launch speed in m/s
g = 9.81  # acceleration due to gravity in m/s^2


def calculate_range_7(t, theta):
    return np.sqrt((u**2) * (t**2) - g * (t**3) * u * np.sin(theta) + 0.25 * (g**2) * (t**4))

def calculate_displacements_7(t, theta):
    x = u * t * np.cos(theta)
    y = u * t * np.sin(theta) - 0.5 * g * t**2
    mask = y >= 0  # filter where y is greater than or equal to 0
    return x[mask], y[mask]

# Generate time values
t = np.linspace(0, 5, 500)

angles = [60, 70, 85, 45]
theta_blue = np.deg2rad(angles[0])
theta_green = np.deg2rad(angles[1])
theta_red = np.deg2rad(angles[2])
theta_yellow = np.deg2rad(angles[3])

r_blue = calculate_range_7(t, theta_blue)
r_green = calculate_range_7(t, theta_green)
r_red = calculate_range_7(t, theta_red)
r_yellow = calculate_range_7(t, theta_yellow)

x_blue, y_blue = calculate_displacements_7(t, theta_blue)
x_green, y_green = calculate_displacements_7(t, theta_green)
x_red, y_red = calculate_displacements_7(t, theta_red)
x_yellow, y_yellow = calculate_displacements_7(t, theta_yellow)

# using formula to calculate min max heights times
def calculate_double_height_times_7(u, g, angles):
    double_height_times = []
    for angle in angles:
        theta = np.deg2rad(angle)
        if np.sin(theta)**2 >= 8/(9 * 9):
            term = np.sqrt(np.sin(theta)**2 - 8/9)
            t1 = (3*u / (2*g)) * (np.sin(theta) + term)
            t2 = (3*u / (2*g)) * (np.sin(theta) - term)
            double_height_times.append((angle, t1, t2))
    return double_height_times

double_height_times = calculate_double_height_times_7(u, g, angles)

# plot range vs time
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=r_blue, mode='lines', name='Angle = 60°', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=t, y=r_green, mode='lines', name='Angle = 70°', line=dict(color='green')))
fig1.add_trace(go.Scatter(x=t, y=r_red, mode='lines', name='Angle = 85°', line=dict(color='red')))
fig1.add_trace(go.Scatter(x=t, y=r_yellow, mode='lines', name='Angle = 45°', line=dict(color='yellow')))

fig1.update_layout(
    title='', #add title here if needed
    xaxis_title='Time (s)',
    yaxis_title='Range (m)',
    margin=dict(l=50, r=50, t=50, b=50),
)

# plot y vs x
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x_blue, y=y_blue, mode='lines', name='Angle = 60°', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=x_green, y=y_green, mode='lines', name='Angle = 70°', line=dict(color='green')))
fig2.add_trace(go.Scatter(x=x_red, y=y_red, mode='lines', name='Angle = 85°', line=dict(color='red')))
fig2.add_trace(go.Scatter(x=x_yellow, y=y_yellow, mode='lines', name='Angle = 45°', line=dict(color='yellow')))

fig2.update_layout(
    title='', #add title here if needed
    xaxis_title='Horizontal Displacement (m)',
    yaxis_title='Vertical Displacement (m)',
    margin=dict(l=50, r=50, t=50, b=50),
)

# Print min max points for X-Y graph (For debugging only)
for angle, t1, t2 in double_height_times:
    theta = np.deg2rad(angle)
    x1, y1 = calculate_displacements_7(t1, theta)
    x2, y2 = calculate_displacements_7(t2, theta)
    
    if len(x1) > 0 and len(y1) > 0 and len(x2) > 0 and len(y2) > 0:
        print(f"Angle = {angle}°, Point 1: ({x1[0]}, {y1[0]}), Point 2: ({x2[0]}, {y2[0]})")

@app.route('/Range-Time-Graph')
def index_7():
    return render_template('task7.html', plot1=fig1.to_json(), plot2=fig2.to_json())


@app.route('/update_7', methods=['POST'])
def update_7():
    global u

    data = request.json
    
    angle_blue = float(data['angle_blue'])
    angle_green = float(data['angle_green'])
    angle_red = float(data['angle_red'])
    angle_yellow = float(data['angle_yellow'])
    u = float(data['speed'])

    theta_blue = np.deg2rad(angle_blue)
    theta_green = np.deg2rad(angle_green)
    theta_red = np.deg2rad(angle_red)
    theta_yellow = np.deg2rad(angle_yellow)

    r_blue_new = calculate_range_7(t, theta_blue)
    r_green_new = calculate_range_7(t, theta_green)
    r_red_new = calculate_range_7(t, theta_red)
    r_yellow_new = calculate_range_7(t, theta_yellow)

    x_blue_new, y_blue_new = calculate_displacements_7(t, theta_blue)
    x_green_new, y_green_new = calculate_displacements_7(t, theta_green)
    x_red_new, y_red_new = calculate_displacements_7(t, theta_red)
    x_yellow_new, y_yellow_new = calculate_displacements_7(t, theta_yellow)

    # Update the corresponding traces based on the angles
    fig1.data[0].y = r_blue_new
    fig1.data[1].y = r_green_new
    fig1.data[2].y = r_red_new
    fig1.data[3].y = r_yellow_new

    fig2.data[0].x = x_blue_new
    fig2.data[0].y = y_blue_new
    fig2.data[1].x = x_green_new
    fig2.data[1].y = y_green_new
    fig2.data[2].x = x_red_new
    fig2.data[2].y = y_red_new
    fig2.data[3].x = x_yellow_new
    fig2.data[3].y = y_yellow_new

    # Clear existing min max points
    fig1.data = [trace for trace in fig1.data if 'Min Max Height' not in trace.name]
    fig2.data = [trace for trace in fig2.data if 'Min Max Height' not in trace.name]

    # Calculate and add new min max points
    double_height_times = calculate_double_height_times_7(u, g, [angle_blue, angle_green, angle_red, angle_yellow])

    for angle, t1, t2 in double_height_times:
        theta = np.deg2rad(angle)
        r1 = calculate_range_7(t1, theta)
        r2 = calculate_range_7(t2, theta)
        x1, y1 = calculate_displacements_7(t1, theta)
        x2, y2 = calculate_displacements_7(t2, theta)
        
        if len(x1) > 0 and len(y1) > 0 and len(x2) > 0 and len(y2) > 0:
            fig1.add_trace(go.Scatter(x=[t1, t2], y=[r1, r2], mode='markers', name=f'Angle = {angle}° Min Max Height', marker=dict(size=10, color='white')))
            fig2.add_trace(go.Scatter(x=[x1[0], x2[0]], y=[y1[0], y2[0]], mode='markers', name=f'Angle = {angle}° Min Max Height', marker=dict(size=10, color='white')))
        
            fig1.update_layout(
                paper_bgcolor='#26252C',  # Background color of the entire paper
                plot_bgcolor='#26252C',   # Background color of the plot area
                font=dict(color='white')  # Font color
            )

            # Update layout for fig2
            fig2.update_layout(
                paper_bgcolor='#26252C',  # Background color of the entire paper
                plot_bgcolor='#26252C',   # Background color of the plot area
                font=dict(color='white')  # Font color
            )

            # Print min max points for X-Y graph
            print(f"Angle = {angle}°, Point 1: ({x1[0]}, {y1[0]}), Point 2: ({x2[0]}, {y2[0]})")

    # Return updated plots JSON
    return jsonify({'plot1': fig1.to_json(), 'plot2': fig2.to_json()})



#Task 9
def compute_trajectory_9(u, theta_deg, h, m, C_d, A):
    theta = np.deg2rad(theta_deg)
    g = 9.81
    rho = 1.225
    dt = 0.01
    
    vx = u * np.cos(theta)
    vy = u * np.sin(theta)
    
    x_data = [0]
    y_data = [h]
    vx_data = [vx]
    vy_data = [vy]
    t_data = [0]
    
    x = 0
    y = h
    t = 0
    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        F_d = 0.5 * C_d * rho * A * v**2
        ax = -F_d * vx / (m * v)
        ay = -g - (F_d * vy / (m * v))

        x_new = x + vx * dt + 0.5 * ax * dt**2
        y_new = y + vy * dt + 0.5 * ay * dt**2
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt

        x = x_new
        y = y_new
        vx = vx_new
        vy = vy_new
        t += dt

        x_data.append(x)
        y_data.append(y)
        vx_data.append(vx)
        vy_data.append(vy)
        t_data.append(t)
    
    x_data_no_drag = [0]
    y_data_no_drag = [h]
    vx_no_drag = u * np.cos(theta)
    vy_no_drag = u * np.sin(theta)
    t = 0
    x = 0
    y = h
    while y >= 0:
        x += vx_no_drag * dt
        y += vy_no_drag * dt - 0.5 * g * dt**2
        vy_no_drag -= g * dt
        t += dt
        x_data_no_drag.append(x)
        y_data_no_drag.append(y)
    
    # Calculate angle based on the formula
    theta_red = np.arcsin(1 / np.sqrt(2 + (2 * g * h) / (u**2)))
    vx_red = u * np.cos(theta_red)
    vy_red = u * np.sin(theta_red)
    
    x_data_red = [0]
    y_data_red = [h]
    t = 0
    x = 0
    y = h
    while y >= 0:
        x += vx_red * dt
        y += vy_red * dt - 0.5 * g * dt**2
        vy_red -= g * dt
        t += dt
        x_data_red.append(x)
        y_data_red.append(y)
    
    return x_data, y_data, x_data_no_drag, y_data_no_drag, x_data_red, y_data_red

@app.route('/page9')
def index_9():
    return render_template('task9.html')

@app.route('/update_9', methods=['POST'])
def update_9():
    u = float(request.json['speed'])
    theta_deg = float(request.json['angle'])
    h = float(request.json['height'])
    m = float(request.json['mass'])
    C_d = float(request.json['drag'])
    A = float(request.json['area'])
    
    x_data, y_data, x_data_no_drag, y_data_no_drag, x_data_red, y_data_red = compute_trajectory_9(u, theta_deg, h, m, C_d, A)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='With air resistance'))
    fig.add_trace(go.Scatter(x=x_data_no_drag, y=y_data_no_drag, mode='lines', name='No air resistance', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=x_data_red, y=y_data_red, mode='lines', name='Red trajectory (Formula angle)', line=dict(color='red')))
    fig.update_layout(title='Projectile Motion with and without Air Resistance',
                      xaxis_title='x /m',
                      yaxis_title='y /m',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'))
    graphJSON = pio.to_json(fig)
    return jsonify(graphJSON)

#Task 8 
@app.route('/COR')
def index_8():
    return render_template('COR.html')


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
