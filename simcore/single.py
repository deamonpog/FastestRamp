"""Single simulation runner (headless or interactive).

This module contains the core physics loop that used to live entirely in
`ramp_sim.py`. The logic is functionally equivalent but organized for reuse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame
import pymunk
import pymunk.pygame_util

from . import constants


@dataclass
class SimulationResult:
    """Container for the output of a single simulation run."""

    time_to_finish: Optional[float]
    physics_log: List[Dict[str, Any]]
    killed: bool
    kill_y: float
    contact_episodes: int
    time_on_ramp: float


def resolve_kill_y(ramp_points: Sequence[Tuple[float, float]], kill_y: Optional[float], kill_offset: float) -> float:
    # Choose an explicit kill line if provided, otherwise place it below the lowest ramp point
    ramp_ys = [p[1] for p in ramp_points]
    lowest_ramp_y = max(ramp_ys)
    return kill_y if kill_y is not None else lowest_ramp_y + kill_offset


def build_space(ramp_points: Sequence[Tuple[float, float]]) -> Tuple[pymunk.Space, List[pymunk.Segment]]:
    # Create the physics space and static ramp segments
    space = pymunk.Space()
    space.gravity = (0, constants.GRAVITY)

    ramps: List[pymunk.Segment] = []
    for i in range(len(ramp_points) - 1):
        p0 = ramp_points[i]
        p1 = ramp_points[i + 1]
        seg = pymunk.Segment(space.static_body, p0, p1, constants.SEGMENT_THICKNESS)
        seg.friction = constants.FRICTION
        seg.collision_type = 2  # RAMP_TYPE
        space.add(seg)
        ramps.append(seg)

    return space, ramps


def create_ball(space: pymunk.Space, start_x: Optional[float], start_y: Optional[float], ramp_points: Sequence[Tuple[float, float]]) -> Tuple[pymunk.Body, pymunk.Circle]:
    # Build the dynamic body and place it either at provided coords or above first ramp
    moment = pymunk.moment_for_circle(constants.MASS, 0, constants.RADIUS)
    ball = pymunk.Body(constants.MASS, moment)

    if (start_x is not None) and (start_y is not None):
        ball.position = (start_x, start_y)
    else:
        ball.position = (ramp_points[0][0], ramp_points[0][1] - 150)

    shape = pymunk.Circle(ball, constants.RADIUS)
    shape.friction = constants.FRICTION
    shape.collision_type = 1  # BALL_TYPE
    space.add(ball, shape)
    return ball, shape


def log_frame(
    physics_log: List[Dict[str, Any]],
    time_elapsed: float,
    ball: pymunk.Body,
    prev_vx: float,
    prev_vy: float,
    on_ramp: bool,
    time_on_ramp: float,
) -> Tuple[float, float]:
    # Capture kinematics and energies for the current frame
    x, y = ball.position
    vx, vy = ball.velocity
    ax = (vx - prev_vx) / constants.DT
    ay = (vy - prev_vy) / constants.DT
    speed = math.sqrt(vx * vx + vy * vy)
    ke = 0.5 * constants.MASS * (speed ** 2)
    pe = constants.MASS * constants.GRAVITY * ((500 - y) / 100)
    te = ke + pe

    physics_log.append({
        "t": time_elapsed,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "ax": ax,
        "ay": ay,
        "speed": speed,
        "KE": ke,
        "PE": pe,
        "TE": te,
        "on_ramp": on_ramp,
        "t_on": time_on_ramp,
    })
    return vx, vy


def run_sim(
    ramp_points: Sequence[Tuple[float, float]],
    start_x: Optional[float] = None,
    start_y: Optional[float] = None,
    show_buttons: bool = True,
    allow_restart: bool = True,
    display: bool = True,
    kill_y: Optional[float] = None,
    kill_offset: float = 250,
) -> SimulationResult:
    """Run a single simulation.

    Parameters mirror the legacy `ramp_sim.run_sim` and the return value is
    unchanged so callers remain compatible.
    """

    kill_line = resolve_kill_y(ramp_points, kill_y, kill_offset)
    # Build world bodies
    space, _ = build_space(ramp_points)
    ball, _ = create_ball(space, start_x, start_y, ramp_points)
    initial_pos = ball.position

    physics_log: List[Dict[str, Any]] = []
    t_start: Optional[float] = None
    time_elapsed = 0.0
    prev_vx, prev_vy = ball.velocity
    on_ramp = False
    current_contacts = 0
    contact_episodes = 0
    time_on_ramp = 0.0

    BALL_TYPE = 1
    RAMP_TYPE = 2

    def begin(_arbiter, _s, _data):
        # Track entry onto ramps and count contact episodes
        nonlocal on_ramp, current_contacts, contact_episodes, t_start
        current_contacts += 1
        if not on_ramp:
            on_ramp = True
            contact_episodes += 1
            if t_start is None:
                t_start = time_elapsed
        return True

    def separate(_arbiter, _s, _data):
        # Track leaving ramp contact
        nonlocal on_ramp, current_contacts
        current_contacts -= 1
        if current_contacts <= 0:
            current_contacts = 0
            on_ramp = False

    space.on_collision(BALL_TYPE, RAMP_TYPE, begin=begin, separate=separate)

    sim_running = (not show_buttons)
    user_edit_mode = show_buttons

    if display:
        # Set up window and draw helpers when visual output is enabled
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

    def restart():
        # Reset simulation state while keeping geometry intact
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

    while True:
        if not display:
            # Headless fast path for multiprocessing runs
            space.step(constants.DT)
            time_elapsed += constants.DT
            if on_ramp:
                time_on_ramp += constants.DT
            prev_vx, prev_vy = log_frame(
                physics_log,
                time_elapsed,
                ball,
                prev_vx,
                prev_vy,
                on_ramp,
                time_on_ramp,
            )
            x, y = ball.position
            if t_start is not None and x > ramp_points[-1][0] + 5:
                # Finished the ramp
                t_total = time_elapsed - t_start
                return SimulationResult(t_total, physics_log, False, kill_line, contact_episodes, time_on_ramp)
            if y > kill_line:
                # Ball fell off-screen
                return SimulationResult(None, physics_log, True, kill_line, contact_episodes, time_on_ramp)
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return SimulationResult(None, None, False, kill_line, contact_episodes, time_on_ramp)  # type: ignore[arg-type]
            if show_buttons and user_edit_mode and event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos):
                    # Start button transitions into running state
                    user_edit_mode = False
                    sim_running = True
            if allow_restart and sim_running and event.type == pygame.MOUSEBUTTONDOWN:
                if restart_button.collidepoint(event.pos):
                    # Restart button rewinds to initial placement
                    restart()

        if sim_running:
            # Advance physics when running interactively
            space.step(constants.DT)
            time_elapsed += constants.DT
            if on_ramp:
                time_on_ramp += constants.DT
            prev_vx, prev_vy = log_frame(
                physics_log,
                time_elapsed,
                ball,
                prev_vx,
                prev_vy,
                on_ramp,
                time_on_ramp,
            )
            x, y = ball.position
            if t_start is not None and x > ramp_points[-1][0] + 5:
                # Finished the ramp
                t_total = time_elapsed - t_start
                pygame.quit()
                return SimulationResult(t_total, physics_log, False, kill_line, contact_episodes, time_on_ramp)
            if y > kill_line:
                # Ball fell off-screen
                pygame.quit()
                return SimulationResult(None, physics_log, True, kill_line, contact_episodes, time_on_ramp)

        screen.fill((25, 25, 25))
        space.debug_draw(draw_options)

        if sim_running:
            # Overlay current state for interactive visualization
            x, y = ball.position
            vx, vy = ball.velocity
            ax = (vx - prev_vx) / constants.DT
            ay = (vy - prev_vy) / constants.DT
            speed = math.sqrt(vx * vx + vy * vy)
            ke = 0.5 * constants.MASS * (speed ** 2)
            pe = constants.MASS * constants.GRAVITY * ((500 - y) / 100)
            te = ke + pe
            overlay_lines = [
                f"t: {time_elapsed:.3f}",
                f"x: {x:.1f}, y: {y:.1f}",
                f"vx: {vx:.2f}, vy: {vy:.2f}",
                f"ax: {ax:.2f}, ay: {ay:.2f}",
                f"speed: {speed:.2f}",
                f"KE: {ke:.2f}, PE: {pe:.2f}",
                f"TE: {te:.2f}",
                f"on_ramp: {on_ramp}",
                f"episodes: {contact_episodes}",
                f"time_on_ramp: {time_on_ramp:.3f}s",
            ]
            for i, line in enumerate(overlay_lines):
                screen.blit(font.render(line, True, (255, 255, 255)), (10, 10 + 22 * i))

        if show_buttons and user_edit_mode:
            # Draw start button when waiting for user
            pygame.draw.rect(screen, (70, 140, 220), start_button)
            screen.blit(font.render("START", True, (255, 255, 255)), (start_button.x + 25, start_button.y + 18))

        if allow_restart and sim_running:
            # Draw restart button during live simulation
            pygame.draw.rect(screen, (220, 70, 70), restart_button)
            screen.blit(font.render("RESTART", True, (255, 255, 255)), (restart_button.x + 10, restart_button.y + 8))

        pygame.display.flip()
        clock.tick(60)
