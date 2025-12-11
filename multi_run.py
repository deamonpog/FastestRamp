# main.py
from ramp_sim import run_multi_parallel, visualize_results_grid

if __name__ == "__main__":
    ramp_points = [
        (100, 100),
        (300, 250),
        (600, 320),
        (800, 800)
    ]

    x_locations = [120, 160, 200, 240, 280, 320, 360, 400]
    y_height = 0

    # 1) Run all sims in parallel, headless
    results = run_multi_parallel(ramp_points, x_locations, y_height, kill_offset=500)

    # 2) Replay all sims in one Pygame window
    kill_y = list(results.values())[0]["kill_y"]
    visualize_results_grid(ramp_points, kill_y, results, window_size=(1200, 800), fps=60)
