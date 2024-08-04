from vpython import sphere, vector, color, textures

earth_sphere = sphere(
    pos=vector(0, 0, 0),
    radius=100,
    texture=textures.earth,
    color=color.white
)

while True:
    pass