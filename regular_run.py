from ramp_sim import run_multi

ramp = [
    (100,350),
    (300,250),
    (600,320),
    (800,300),
]

x_positions = [120, 160, 200, 260, 320]
y_height = 80

results = run_multi(ramp, x_positions, y_height, display=True)

for x, data in results.items():
    print(f"x={x} -> time={data['t']:.4f} s")
