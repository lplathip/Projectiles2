from vpython import sphere, vector, color, textures, scene, rate, curve, mag, label, arrow, radians, sin, cos
# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M = 5.972e24  # Mass of Earth in kg
earth_radius = 6.371e6  # Radius of Earth in meters
omega = 7.2921e-5  # Angular velocity of Earth's rotation in rad/s

# Create Earth with texture
earth = sphere(pos=vector(0, 0, 0), radius=earth_radius, texture=textures.earth)

# Convert latitude and longitude to Cartesian coordinates
latitude = 52.36  # in degrees
longitude = 1.17  # in degrees

# Convert degrees to radians
phi = radians(latitude)
lambda_ = radians(longitude)

# Calculate Cartesian coordinates
z = earth_radius * cos(phi) * cos(lambda_)
x = earth_radius * cos(phi) * sin(lambda_)
y = earth_radius * sin(phi)

# Create a projectile
projectile_radius = earth_radius / 100
projectile = sphere(pos=vector(x, y, z), radius=projectile_radius, color=color.red)

# Create labeled axes
axis_length = earth_radius * 1.5
shaftwidth = earth_radius * 0.01  # Thinner axis width
x_axis = arrow(pos=vector(0, 0, 0), axis=vector(axis_length, 0, 0), color=color.blue, shaftwidth=shaftwidth)
y_axis = arrow(pos=vector(0, 0, 0), axis=vector(0, axis_length, 0), color=color.green, shaftwidth=shaftwidth)
z_axis = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, axis_length), color=color.yellow, shaftwidth=shaftwidth)

label(pos=x_axis.pos + x_axis.axis, text='x', xoffset=10, yoffset=10, space=30, height=20, border=4, font='sans')
label(pos=y_axis.pos + y_axis.axis, text='y', xoffset=10, yoffset=10, space=30, height=20, border=4, font='sans')
label(pos=z_axis.pos + z_axis.axis, text='z', xoffset=10, yoffset=10, space=30, height=20, border=4, font='sans')

vx, vy, vz = 0.5953, 0.3304, 0.6014

# Initial velocity of the projectile
speed = 5500  # m/s
direction = vector(vx, vy, vz).norm()  # Normalize the direction vector
velocity = direction * speed  # Velocity vector

# Time step for the simulation
dt = 0.1  # seconds

# Create a curve object for the tracer
tracer = curve(color=color.yellow)

# Tilt angle in degrees and convert to radians
tilt_angle = radians(23.5)
rotation_axis = vector(sin(tilt_angle), cos(tilt_angle), 0)

# Keep the window open and update the projectile's position
counter = 0
while True:
    rate(1000)  # Limit the loop to 100 iterations per second
    
    # Rotate the Earth around the tilted axis
    earth.rotate(angle=omega * dt, axis=rotation_axis)
    
    # Calculate the distance from the center of the Earth to the projectile
    r = projectile.pos - earth.pos  # Displacement vector from the center of the Earth to the projectile
    r_mag = mag(r)  # Magnitude of the displacement vector
    
    # Calculate the gravitational acceleration
    g = G * M / r_mag**2
    gravitational_acceleration = -g * r.norm()  # Gravitational acceleration vector (directed towards the center of the Earth)
    
    # Calculate the centrifugal acceleration
    centrifugal_acceleration = omega**2 * r_mag * r.norm()  # Centrifugal acceleration vector (outward from the axis of rotation)
    
    # Net acceleration considering both gravity and centrifugal force
    net_acceleration = gravitational_acceleration + centrifugal_acceleration
    
    # Update the projectile's velocity
    velocity += net_acceleration * dt
    
    # Update the projectile's position
    projectile.pos += velocity * dt
    
    # Add the new position to the tracer
    tracer.append(pos=projectile.pos)
    
    counter += 1

    if counter > 2 and r_mag < earth_radius:
        velocity = vector(0, 0 ,0)

