"""Core simulation constants.

These defaults are shared across headless and interactive runs.
"""

GRAVITY = 900             # Downward gravity used by pymunk (pixels/s^2)
RADIUS = 20               # Ball radius in pixels
MASS = 1                  # Ball mass in arbitrary units
DT = 1 / 60               # Fixed physics timestep
FRICTION = 0.8            # Friction coefficient shared by ramps and ball
SEGMENT_THICKNESS = 5     # Ramp segment thickness for pymunk segments
