from vpython import sphere, vector, color, textures, scene, rate, curve, mag
from math import radians, sin, cos  # Import the necessary functions

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M = 5.972e24  # Mass of Earth in kg
earth_radius = 6.371e6  # Radius of Earth in meters
omega = 7.2921e-5  # Angular velocity of Earth's rotation in rad/s

# Create Earth with texture
earth = sphere(pos=vector(0, 0, 0), radius=earth_radius, texture=textures.earth)

# Convert latitude and longitude to Cartesian coordinates
latitude = 16  # in degrees
longitude = 100  # in degrees

# Convert degrees to radians
phi = radians(latitude)
lambda_ = radians(longitude)

# Calculate Cartesian coordinates
z = earth_radius * cos(phi) * cos(lambda_)
x = earth_radius * cos(phi) * sin(lambda_)
y = earth_radius * sin(phi)

# Create a projectile
projectile_radius = earth_radius / 100
projectile = sphere(pos=vector(x, y , z), radius=projectile_radius, color=color.red)

ballX = sphere(pos=vector(earth_radius, 0 , 0), radius=projectile_radius, color=color.blue)
ballY = sphere(pos=vector(0, earth_radius , 0), radius=projectile_radius, color=color.green)
ballZ = sphere(pos=vector(0, 0 , earth_radius), radius=projectile_radius, color=color.yellow)


vx, vy, vz = 0.5953,0.3304,0.7314

# Initial velocity of the projectile
speed = 6000  # m/s
direction = vector(vx, vy, vz).norm()  # Normalize the direction vector
velocity = direction * speed  # Velocity vector

# Time step for the simulation
dt = 1  # seconds

# Create a curve object for the tracer
tracer = curve(color=color.yellow)

# Tilt angle in degrees and convert to radians
tilt_angle = radians(23.5)
rotation_axis = vector(sin(tilt_angle), cos(tilt_angle), 0)

# Keep the window open and update the projectile's position
while True:
    rate(100)  # Limit the loop to 100 iterations per second
    
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
