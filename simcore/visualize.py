"""Replay multiple simulation logs in a grid window."""
from __future__ import annotations

import math
import pygame


def visualize_results_grid(
    ramp_points,
    kill_y,
    results,
    window_size=(1200, 800),
    fps=60,
):
    """Visualize multiple simulation results in a single Pygame window."""

    pygame.init()
    W, H = window_size
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Ramp Simulations - Grid Replay")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 16)

    sims = sorted(results.items(), key=lambda kv: kv[0])
    n_sims = len(sims)
    if n_sims == 0:
        print("No simulations to visualize.")
        return

    # Pick a near-square grid to pack all panels
    cols = math.ceil(math.sqrt(n_sims))
    rows = math.ceil(n_sims / cols)

    xs = [p[0] for p in ramp_points]
    ys = [p[1] for p in ramp_points] + [kill_y]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    world_w = max_x - min_x if max_x > min_x else 1
    world_h = max_y - min_y if max_y > min_y else 1

    cell_w = W / cols
    cell_h = H / rows
    margin = 18

    log_lists = [data["log"] for _, data in sims]
    max_len = max(len(log) for log in log_lists)

    def world_to_cell(ix, x, y):
        # Map world coordinates into the ix-th grid cell in screen space
        col = ix % cols
        row = ix // cols
        offset_x = col * cell_w
        offset_y = row * cell_h
        scale_x = (cell_w - 2 * margin) / world_w
        scale_y = (cell_h - 2 * margin) / world_h
        scale = min(scale_x, scale_y)
        sx = offset_x + margin + (x - min_x) * scale
        sy = offset_y + margin + (y - min_y) * scale
        return int(sx), int(sy)

    frame = 0
    playing = True
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Toggle playback pause
                    playing = not playing
                elif event.key == pygame.K_r:
                    # Restart replay from frame zero
                    frame = 0
                    playing = True

        if playing:
            frame += 1
            if frame >= max_len:
                frame = max_len - 1
                playing = False

        frame = max(0, min(frame, max_len - 1))

        screen.fill((10, 10, 10))

        for idx, (x_start, data) in enumerate(sims):
            log = data["log"]
            killed = data["killed"]
            t_finish = data["time"]

            col = idx % cols
            row = idx // cols
            cell_rect = pygame.Rect(int(col * cell_w), int(row * cell_h), int(cell_w), int(cell_h))

            # Panel background and border
            pygame.draw.rect(screen, (20, 20, 20), cell_rect, 0)
            pygame.draw.rect(screen, (60, 60, 60), cell_rect, 1)

            # Draw the ramp polyline inside this panel
            for i in range(len(ramp_points) - 1):
                x0, y0 = ramp_points[i]
                x1, y1 = ramp_points[i + 1]
                sx0, sy0 = world_to_cell(idx, x0, y0)
                sx1, sy1 = world_to_cell(idx, x1, y1)
                pygame.draw.line(screen, (150, 150, 150), (sx0, sy0), (sx1, sy1), 2)

            kx0, ky0 = world_to_cell(idx, min_x, kill_y)
            kx1, ky1 = world_to_cell(idx, max_x, kill_y)
            # Horizontal kill line reference
            pygame.draw.line(screen, (255, 80, 80), (kx0, ky0), (kx1, ky1), 1)
            kill_label = font.render("kill_y", True, (255, 80, 80))
            screen.blit(kill_label, (kx0 + 3, ky0 - 10))

            if len(log) > 0:
                local_idx = min(frame, len(log) - 1)
                state = log[local_idx]
                bx, by = state["x"], state["y"]
                sx, sy = world_to_cell(idx, bx, by)
                # Ball marker: green if alive, red if killed
                color = (0, 200, 0) if not killed else (200, 50, 50)
                pygame.draw.circle(screen, color, (sx, sy), 5)

                overlay_lines = [
                    f"t={state['t']:.3f}s  t_on={state.get('t_on', 0.0):.3f}s",
                    f"x={state['x']:.1f}, y={state['y']:.1f}",
                    f"vx={state['vx']:.2f}, vy={state['vy']:.2f}",
                    f"ax={state['ax']:.2f}, ay={state['ay']:.2f}",
                    f"KE={state['KE']:.1f}, PE={state['PE']:.1f}",
                ]
                for j, line in enumerate(overlay_lines):
                    text = font.render(line, True, (220, 220, 220))
                    tx = text.get_width()
                    x_pos = cell_rect.right - 5 - tx
                    y_pos = cell_rect.y + 40 + 14 * j
                    screen.blit(text, (x_pos, y_pos))

            episodes = data.get("contact_episodes", 0)
            t_on = data.get("time_on_ramp") or 0.0

            label1 = f"x={x_start}  ep={episodes}"
            if killed:
                # Show kill outcome and time-on-ramp
                label2 = f"killed  t_on={t_on:.3f}s"
                label2_color = (255, 80, 80)
            else:
                if t_finish is not None:
                    label2 = f"t={t_finish:.3f}s  t_on={t_on:.3f}s"
                else:
                    label2 = f"t=?  t_on={t_on:.3f}s"
                label2_color = (255, 255, 0)

            text1 = font.render(label1, True, (255, 255, 255))
            text2 = font.render(label2, True, label2_color)
            screen.blit(text1, (cell_rect.right - 5 - text1.get_width(), cell_rect.y + 5))
            screen.blit(text2, (cell_rect.right - 5 - text2.get_width(), cell_rect.y + 22))

        # Global UI hint across all panels
        hint = "SPACE: pause/resume   R: replay   ESC: quit"
        hint_text = font.render(hint, True, (200, 200, 200))
        screen.blit(hint_text, (10, H - 25))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
