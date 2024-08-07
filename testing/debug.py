from vpython import *

# Create two spheres
sphere1 = sphere(pos=vector(-5,0,0), radius=1, color=color.red, make_trail=True)
sphere2 = sphere(pos=vector(5,0,0), radius=1, color=color.blue, make_trail=True)

# Set initial velocities
sphere1.velocity = vector(1,0,0)
sphere2.velocity = vector(-1,0,0)

# Set the time step
dt = 0.01

# Simulation loop
while True:
    rate(100)
    
    # Update positions
    sphere1.pos += sphere1.velocity * dt
    sphere2.pos += sphere2.velocity * dt
    
    # Check for collision
    if mag(sphere1.pos - sphere2.pos) < sphere1.radius + sphere2.radius:
        # Set both velocities to 0 on collision
        sphere1.velocity = vector(0,0,0)
        sphere2.velocity = vector(0,0,0)
        
