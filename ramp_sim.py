import pygame
import pymunk
import pymunk.pygame_util
import math
from multiprocessing import Pool, cpu_count

# ---------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------
GRAVITY = 900
RADIUS = 20
MASS = 1
DT = 1/60


# ==========================================================
# CORE SIMULATION (SINGLE BALL)
# ==========================================================
def run_sim(
    ramp_points,
    start_x=None,
    start_y=None,
    show_buttons=True,
    allow_restart=True,
    display=True,
    kill_y=None,
    kill_offset=250
):
    """
    Run a single simulation.

    Parameters
    ----------
    ramp_points : list[(x,y)]
    start_x     : float or None
    start_y     : float or None
    show_buttons: bool
    allow_restart: bool
    display     : bool
    kill_y      : float or None   # Explicit kill line
    kill_offset : float           # Offset if kill_y not provided

    Returns
    -------
    (time_to_finish, physics_log, killed, kill_y, contact_episodes, time_on_ramp) : tuple
        time_to_finish: float or None
            Time from first contact to passing the end of the last ramp (if reached).
        physics_log   : list of dict physics trace
            Per-frame physics info (x, y, v, a, energies, on_ramp).
        killed        : True/False did the ball fall off
            True if ball fell below kill_y.
        kill_y        : float
            The y-coordinate of the kill line used in this sim.
        contact_episodes : int
            Number of separate on-ramp intervals (off→on transitions).
        time_on_ramp     : float
            Total simulated time during which the ball was in contact with any ramp.
    """

    # ------------------------------------------
    # DETERMINE KILL LINE
    # ------------------------------------------
    ramp_ys = [p[1] for p in ramp_points]
    lowest_ramp_y = max(ramp_ys)

    if kill_y is None:
        kill_y = lowest_ramp_y + kill_offset

    # ------------------------------------------
    # PHYSICS WORLD
    # ------------------------------------------
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)

    # Collision types
    BALL_TYPE = 1
    RAMP_TYPE = 2

    # Ramps
    ramps = []
    for i in range(len(ramp_points) - 1):
        p0 = ramp_points[i]
        p1 = ramp_points[i + 1]
        seg = pymunk.Segment(space.static_body, p0, p1, 5)
        seg.friction = 0.8
        seg.collision_type = RAMP_TYPE
        space.add(seg)
        ramps.append(seg)

    # Ball
    moment = pymunk.moment_for_circle(MASS, 0, RADIUS)
    ball = pymunk.Body(MASS, moment)

    if (start_x is not None) and (start_y is not None):
        ball.position = (start_x, start_y)
    else:
        # Default spawn above first ramp
        ball.position = (ramp_points[0][0], ramp_points[0][1] - 150)

    initial_pos = ball.position  # for safe restart

    shape = pymunk.Circle(ball, RADIUS)
    shape.friction = 0.8
    shape.collision_type = BALL_TYPE
    space.add(ball, shape)

    # ------------------------------------------
    # CONTACT TRACKING
    # ------------------------------------------
    physics_log = []
    t_start = None # first time we ever get on_ramp
    # contact_started = False
    time_elapsed = 0.0

    # for acceleration (numerical derivative)
    prev_vx, prev_vy = ball.velocity

    # on/off ramp state
    on_ramp = False
    current_contacts = 0      # number of overlapping contacts (segments)
    contact_episodes = 0      # off→on transitions
    time_on_ramp = 0.0        # total time with on_ramp=True

    def begin(arbiter, s, data):
        nonlocal on_ramp, current_contacts, contact_episodes, t_start
        current_contacts += 1
        if not on_ramp:
            on_ramp = True
            contact_episodes += 1
            if t_start is None:  # first ever contact
                t_start = time_elapsed
        return True  # continue normal processing

    def separate(arbiter, s, data):
        nonlocal on_ramp, current_contacts
        current_contacts -= 1
        if current_contacts <= 0:
            current_contacts = 0
            on_ramp = False
        # no return needed for separate

    # Collision handler ball <-> ramp
    handler = space.on_collision(BALL_TYPE, RAMP_TYPE, begin = begin, separate = separate)

    # ------------------------------------------
    # GUI SETUP (if display)
    # ------------------------------------------
    sim_running = (not show_buttons)
    user_edit_mode = show_buttons

    # ------------------------------------------
    # PYGAME (IF DISPLAY)
    # ------------------------------------------
    if display:
        pygame.init()
        WIDTH, HEIGHT = 900, 500
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ramp Simulation")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 28)
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        if show_buttons:
            start_button = pygame.Rect(380, 200, 140, 60)
        if allow_restart:
            restart_button = pygame.Rect(760, 20, 120, 40)

    # ------------------------------------------
    # RESTART FUNCTION
    # ------------------------------------------
    def restart():
        nonlocal sim_running, user_edit_mode, time_elapsed, prev_vx, prev_vy
        nonlocal on_ramp, current_contacts, contact_episodes, time_on_ramp, t_start
        ball.position = initial_pos
        ball.velocity = (0, 0)
        time_elapsed = 0.0
        prev_vx, prev_vy = ball.velocity
        on_ramp = False
        current_contacts = 0
        contact_episodes = 0
        time_on_ramp = 0.0
        t_start = None
        sim_running = False
        user_edit_mode = True
        physics_log.clear()

    # =======================================================
    # MAIN LOOP
    # =======================================================
    while True:

        # =========== NO DISPLAY (FAST HEADLESS MODE) ===========
        if not display:
            space.step(DT)
            time_elapsed += DT

            if on_ramp:
                time_on_ramp += DT

            x, y = ball.position
            vx, vy = ball.velocity
            ax = (vx - prev_vx) / DT
            ay = (vy - prev_vy) / DT
            prev_vx, prev_vy = vx, vy

            speed = math.sqrt(vx * vx + vy * vy)
            KE = 0.5 * MASS * (speed ** 2)
            PE = MASS * GRAVITY * ((500 - y) / 100)  # purely for visualization units
            TE = KE + PE

            physics_log.append({
                "t": time_elapsed,
                "x": x, "y": y,
                "vx": vx, "vy": vy,
                "ax": ax, "ay": ay,
                "speed": speed,
                "KE": KE, "PE": PE, "TE": TE,
                "on_ramp": on_ramp,
            })

            # end-of-ramp detection (same as before, but doesn't define on/off)

            # End reached?
            if t_start is not None and x > ramp_points[-1][0] + 5:
                t_total = time_elapsed - t_start
                return t_total, physics_log, False, kill_y, contact_episodes, time_on_ramp

            # Kill condition
            if y > kill_y:
                return None, physics_log, True, kill_y, contact_episodes, time_on_ramp

            continue

        # =========== DISPLAY MODE EVENTS ===========
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None, False, kill_y, contact_episodes, time_on_ramp

            if show_buttons and user_edit_mode and event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos):
                    user_edit_mode = False
                    sim_running = True

            if allow_restart and sim_running and event.type == pygame.MOUSEBUTTONDOWN:
                if restart_button.collidepoint(event.pos):
                    restart()

        # =========== DISPLAY MODE PHYSICS ===========
        if sim_running:
            space.step(DT)
            time_elapsed += DT

            if on_ramp:
                time_on_ramp += DT

            x, y = ball.position
            vx, vy = ball.velocity
            ax = (vx - prev_vx) / DT
            ay = (vy - prev_vy) / DT
            prev_vx, prev_vy = vx, vy

            speed = math.sqrt(vx * vx + vy * vy)
            KE = 0.5 * MASS * (speed ** 2)
            PE = MASS * GRAVITY * ((500 - y) / 100)
            TE = KE + PE

            physics_log.append({
                "t": time_elapsed,
                "x": x, "y": y,
                "vx": vx, "vy": vy,
                "ax": ax, "ay": ay,
                "speed": speed,
                "KE": KE, "PE": PE, "TE": TE,
                "on_ramp": on_ramp,
            })
            # end-of-ramp detection

            # End reached
            if t_start is not None and x > ramp_points[-1][0] + 5:
                t_total = time_elapsed - t_start
                pygame.quit()
                return t_total, physics_log, False, kill_y, contact_episodes, time_on_ramp

            # Kill condition
            if y > kill_y:
                pygame.quit()
                return None, physics_log, True, kill_y, contact_episodes, time_on_ramp

        # =========== DISPLAY RENDER ===========
        screen.fill((25, 25, 25))
        space.debug_draw(draw_options)

        # overlays
        if sim_running:
            overlay_lines = [
                f"t: {time_elapsed:.3f}",
                f"x: {x:.1f}, y: {y:.1f}",
                f"vx: {vx:.2f}, vy: {vy:.2f}",
                f"ax: {ax:.2f}, ay: {ay:.2f}",
                f"speed: {speed:.2f}",
                f"KE: {KE:.2f}, PE: {PE:.2f}",
                f"TE: {TE:.2f}",
                f"on_ramp: {on_ramp}",
                f"episodes: {contact_episodes}",
                f"time_on_ramp: {time_on_ramp:.3f}s",
            ]
            for i, line in enumerate(overlay_lines):
                screen.blit(font.render(line, True, (255, 255, 255)), (10, 10 + 22 * i))

        # Start button
        if show_buttons and user_edit_mode:
            pygame.draw.rect(screen, (70, 140, 220), start_button)
            screen.blit(font.render("START", True, (255, 255, 255)),
                        (start_button.x + 25, start_button.y + 18))

        # Restart button
        if allow_restart and sim_running:
            pygame.draw.rect(screen, (220, 70, 70), restart_button)
            screen.blit(font.render("RESTART", True, (255, 255, 255)),
                        (restart_button.x + 10, restart_button.y + 8))

        pygame.display.flip()
        clock.tick(60)

# END run_sim


# ==========================================================
# MULTI-RUN WRAPPER
# ==========================================================
def run_multi(ramp_points, x_locations, y_height, display=False, kill_offset=250):
    """
    Run a simulation for each starting X-location.
    Returns a dictionary of results for each X.
    """

    results = {}

    for x in x_locations:
        print(f"\n=== Sim from (x={x}, y={y_height}) ===")

        t, log, killed, kill_y, contact_episodes, time_on_ramp = run_sim(
            ramp_points,
            start_x=x,
            start_y=y_height,
            show_buttons=False,
            allow_restart=False,
            display=display,
            kill_y=None,
            kill_offset=kill_offset
        )

        results[x] = {
            "time": t,
            "log": log,
            "killed": killed,
            "kill_y": kill_y,
            "contact_episodes": contact_episodes,
            "time_on_ramp": time_on_ramp
        }

    return results

# ==========================================================
# MULTI-RUN PARALLEL (HEADLESS) WRAPPER
# ==========================================================

def _sim_worker(args):
    """
    Worker function for a single headless simulation.
    This runs in a separate process (no pygame, display=False).
    """
    ramp_points, x, y_height, kill_offset = args
    t, log, killed, kill_y, contact_episodes, time_on_ramp = run_sim(
        ramp_points,
        start_x=x,
        start_y=y_height,
        show_buttons=False,
        allow_restart=False,
        display=False,      # IMPORTANT: headless for multiprocessing
        kill_y=None,
        kill_offset=kill_offset
    )
    return x, {
        "time": t, 
        "log": log, 
        "killed": killed, 
        "kill_y": kill_y, 
        "contact_episodes": contact_episodes, 
        "time_on_ramp": time_on_ramp
        }


def run_multi_parallel(ramp_points, x_locations, y_height, kill_offset=250):
    """
    Run multiple simulations in parallel (headless).
    Returns a dict: {x_start: {"time": t, "log": log, "killed": bool}, ...}
    """
    args = [(ramp_points, x, y_height, kill_offset) for x in x_locations]
    n_cpus = cpu_count()
    print(f"Running {len(x_locations)} simulations on {n_cpus} cores (headless)...")

    results = {}
    with Pool(processes=n_cpus) as pool:
        for x, data in pool.map(_sim_worker, args):
            results[x] = data

    return results


def visualize_results_grid(
    ramp_points,
    kill_y,
    results,
    window_size=(1200, 800),
    fps=60
):
    """
    Visualize multiple simulation results in a single Pygame window
    arranged as a grid of small viewports.

    Controls:
      - SPACE : pause / resume
      - R     : restart / replay from the beginning
      - ESC or window close: exit

    Parameters
    ----------
    ramp_points : list[(x,y)]
        Geometry of the ramps (same used for sim).
    results : dict
        {x_start: {"time": t, "log": log, "killed": bool}, ...}
    window_size : (w,h)
    fps : int
        Replay speed.
    """

    pygame.init()
    W, H = window_size
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Ramp Simulations - Grid Replay")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 16)

    # Turn results dict into a list to keep ordering
    sims = sorted(results.items(), key=lambda kv: kv[0])  # sort by x_start
    n_sims = len(sims)

    if n_sims == 0:
        print("No simulations to visualize.")
        return

    # Determine grid layout (roughly square)
    cols = math.ceil(math.sqrt(n_sims))
    rows = math.ceil(n_sims / cols)

    # Compute bounding box for ramps (+ kill line) to scale into each cell
    xs = [p[0] for p in ramp_points]
    ys = [p[1] for p in ramp_points] + [kill_y]  # include kill_y so line is visible

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Some padding in "world" units
    world_w = max_x - min_x if max_x > min_x else 1
    world_h = max_y - min_y if max_y > min_y else 1

    cell_w = W / cols
    cell_h = H / rows
    margin = 18

    # Precompute max log length to know when to stop replay
    log_lists = [data["log"] for _, data in sims]
    max_len = max(len(log) for log in log_lists)

    def world_to_cell(ix, x, y):
        """
        Transform world coordinates (x,y) into pixel coords
        of cell index ix in the grid.
        """
        col = ix % cols
        row = ix // cols
        offset_x = col * cell_w
        offset_y = row * cell_h

        # Scale to fit inside cell with margins
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
        # --------- EVENT HANDLING ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Pause / resume
                    playing = not playing
                elif event.key == pygame.K_r:
                    # Restart / replay
                    frame = 0
                    playing = True

        # Advance frame if playing
        if playing:
            frame += 1
            if frame >= max_len:
                # Reached end: stop and wait for R or SPACE
                frame = max_len - 1
                playing = False

        # Clamp frame to valid range
        frame = max(0, min(frame, max_len - 1))

        # --------- DRAW ----------
        screen.fill((10, 10, 10))

        # For each simulation, draw ramps + ball + kill line + overlays
        for idx, (x_start, data) in enumerate(sims):
            log = data["log"]
            killed = data["killed"]
            t_finish = data["time"]

            # Cell rect
            col = idx % cols
            row = idx // cols
            cell_rect = pygame.Rect(
                int(col * cell_w), int(row * cell_h),
                int(cell_w), int(cell_h)
            )

            # Cell background + border
            pygame.draw.rect(screen, (20, 20, 20), cell_rect, 0)
            pygame.draw.rect(screen, (60, 60, 60), cell_rect, 1)

            # Draw ramps in this cell
            for i in range(len(ramp_points) - 1):
                x0, y0 = ramp_points[i]
                x1, y1 = ramp_points[i + 1]
                sx0, sy0 = world_to_cell(idx, x0, y0)
                sx1, sy1 = world_to_cell(idx, x1, y1)
                pygame.draw.line(screen, (150, 150, 150), (sx0, sy0), (sx1, sy1), 2)

            # ---------- draw kill_y line in this panel ----------
            kx0, ky0 = world_to_cell(idx, min_x, kill_y)
            kx1, ky1 = world_to_cell(idx, max_x, kill_y)
            pygame.draw.line(screen, (255, 80, 80), (kx0, ky0), (kx1, ky1), 1)
            # small label
            kill_label = font.render("kill_y", True, (255, 80, 80))
            screen.blit(kill_label, (kx0 + 3, ky0 - 10))

            # ---------- ball + per-panel overlay ----------
            if len(log) > 0:
                local_idx = min(frame, len(log) - 1)
                state = log[local_idx]
                bx, by = state["x"], state["y"]
                sx, sy = world_to_cell(idx, bx, by)

                # Draw ball
                color = (0, 200, 0) if not killed else (200, 50, 50)
                pygame.draw.circle(screen, color, (sx, sy), 5)

                # Per-cell overlay: pos, vel, acc, energy
                overlay_lines = [
                    f"x={state['x']:.1f}, y={state['y']:.1f}",
                    f"vx={state['vx']:.2f}, vy={state['vy']:.2f}",
                    f"ax={state['ax']:.2f}, ay={state['ay']:.2f}",
                    f"KE={state['KE']:.1f}, PE={state['PE']:.1f}",
                ]
                for j, line in enumerate(overlay_lines):
                    text = font.render(line, True, (220, 220, 220))
                    tx = text.get_width()
                    # Right aligned inside this panel
                    x_pos = cell_rect.right - 5 - tx
                    y_pos = cell_rect.y + 40 + 14 * j
                    screen.blit(text, (x_pos, y_pos))

            # # Labels: x start + status
            # label1 = f"x={x_start}"
            # if killed:
            #     label2 = "killed"
            #     label2_color = (255, 80, 80)
            # else:
            #     if t_finish is not None:
            #         label2 = f"t={t_finish:.3f}s"
            #     else:
            #         label2 = "t=?"
            #     label2_color = (255, 255, 0)

            # text1 = font.render(label1, True, (255, 255, 255))
            # text2 = font.render(label2, True, label2_color)
            # screen.blit(text1, (cell_rect.x + 5, cell_rect.y + 5))
            # screen.blit(text2, (cell_rect.x + 5, cell_rect.y + 22))
            episodes = data["contact_episodes"]
            t_on = data["time_on_ramp"] or 0.0

            # Line 1: start x + episodes
            label1 = f"x={x_start}  ep={episodes}"

            # Line 2: finish / killed + time_on_ramp
            if killed:
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
            screen.blit(text1, (cell_rect.x + 5, cell_rect.y + 5))
            screen.blit(text2, (cell_rect.x + 5, cell_rect.y + 22))


        # UI hint at bottom
        hint = "SPACE: pause/resume   R: replay   ESC: quit"
        hint_text = font.render(hint, True, (200, 200, 200))
        screen.blit(hint_text, (10, H - 25))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()

# END visualize_results_grid