from vpython import *
import math

# Set up the scene
scene = canvas(title='Projectile on Earth', width=800, height=600, center=vector(0,0,0), background=color.black)

# Create Earth
earth_radius = 6.371e6  # in meters
earth = sphere(pos=vector(0,0,0), radius=earth_radius, texture=textures.earth)

# Function to convert latitude and longitude to XYZ coordinates
def latlon_to_xyz(lat, lon, radius):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
    z = radius * math.sin(lat_rad)
    return vector(x, y, z)

# Function to convert azimuth and elevation to velocity vector
def azimuth_elevation_to_velocity(azimuth, elevation, speed):
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)
    vx = speed * math.cos(elevation_rad) * math.cos(azimuth_rad)
    vy = speed * math.cos(elevation_rad) * math.sin(azimuth_rad)
    vz = speed * math.sin(elevation_rad)
    return vector(vx, vy, vz)

# Specific inputs
latitude = 15  # degrees
longitude = 1  # degrees
azimuth = 30  # degrees from north
elevation = 30  # degrees from horizontal
speed = 3000  # m/s

# Convert latitude and longitude to initial position
proj_initial_pos = latlon_to_xyz(latitude, longitude, earth_radius + 10)  # 10 meters above the surface

# Convert azimuth and elevation to initial velocity
proj_velocity = azimuth_elevation_to_velocity(azimuth, elevation, speed)

# Create the projectile
proj_radius = earth_radius / 100  # 1/100th of the Earth's radius
projectile = sphere(pos=proj_initial_pos, radius=proj_radius, color=color.red, make_trail=True)

# Gravity
g = 9.81

# Time step
dt = 0.01

# Simulation loop
while projectile.pos.mag < earth_radius + 5000:  # stop when the projectile is far from the Earth
    rate(100)
    proj_velocity.z -= g * dt  # Update velocity due to gravity
    projectile.pos += proj_velocity * dt  # Update position
