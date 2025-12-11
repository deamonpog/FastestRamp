from ramp_sim import run_multi

# Example sequential run with interactive display

ramp = [
    (100,350),
    (300,250),
    (600,320),
    (800,300),
]

x_positions = [120, 160, 200, 260, 320]
y_height = 80

# Run and render each start position one after another
results = run_multi(ramp, x_positions, y_height, display=True)

# Print summarized finish times
for x, data in results.items():
    print(f"x={x} -> time={data['t']:.4f} s")
