"""Top-down Pygame view of the ring, plus the keyboard controller.

Pygame is only touched in this module, so the simulation and headless training
never depend on a display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from boxing_ai.boxer import Boxer, BoxerCommand, GuardState
from boxing_ai.config import GameConfig
from boxing_ai.geometry import heading_vector, world_to_local
from boxing_ai.simulation import BoxingMatch
from boxing_ai.strikes import Hand, StrikePhase, StrikeType

# ---------------------------------------------------------------- layout (px)
WINDOW_WIDTH = 720
HUD_HEIGHT = 170
RING_PIXELS = 560
RING_MARGIN = (WINDOW_WIDTH - RING_PIXELS) // 2
WINDOW_HEIGHT = HUD_HEIGHT + RING_PIXELS + 2 * RING_MARGIN
BAR_WIDTH = 280
BAR_HEIGHT = 16
HUD_PADDING = 20
LINE_HEIGHT = 22
FONT_SIZE = 24
MESSAGE_FONT_SIZE = 36
ROPE_THICKNESS = 4
FACING_LINE_THICKNESS = 3
OUTLINE_THICKNESS = 2

# --------------------------------------------------------------------- colours
BACKGROUND = (24, 26, 32)
CANVAS = (58, 96, 128)
ROPE = (230, 230, 230)
CENTER_MARK = (90, 130, 160)
TEXT = (235, 235, 235)
BAR_BACKGROUND = (60, 60, 70)
HEALTH_COLOR = (214, 64, 64)
STAMINA_COLOR = (240, 196, 60)
EXHAUSTED_TINT = (120, 120, 120)
HEAD_COLOR = (238, 200, 170)
GLOVE_REST = (200, 40, 40)
GLOVE_WINDUP = (255, 200, 0)
GLOVE_ACTIVE = (255, 255, 255)
GUARD_OUTLINE = (120, 255, 160)
MESSAGE_SHADE = (0, 0, 0, 170)
BOXER_COLORS = ((70, 130, 230), (230, 110, 60))
BOXER_LABELS = ("AI BOXER", "PLAYER")

GUARD_DESCRIPTIONS = {
    GuardState.NONE: "ready",
    GuardState.HIGH: "high guard",
    GuardState.LOW: "low guard",
    GuardState.SLIP: "slipping",
}

GLOVE_PHASE_COLORS = {
    StrikePhase.WINDUP: GLOVE_WINDUP,
    StrikePhase.ACTIVE: GLOVE_ACTIVE,
    StrikePhase.RECOVERY: GLOVE_REST,
}


@dataclass(frozen=True)
class HudInfo:
    """Text shown above the ring."""

    status_text: str
    episode_reward: float
    last_reward: float


class Renderer:
    """Draws a match either into a window ("human") or an off-screen surface."""

    def __init__(self, config: GameConfig, show_window: bool) -> None:
        self._config = config
        self._show_window = show_window
        self._pixels_per_metre = RING_PIXELS / (2.0 * config.ring.half_size)
        pygame.init()
        if show_window:
            pygame.display.set_caption("Boxing AI - sparring")
            self._surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        else:
            self._surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self._font = pygame.font.Font(None, FONT_SIZE)
        self._message_font = pygame.font.Font(None, MESSAGE_FONT_SIZE)
        self._clock = pygame.time.Clock()

    # ------------------------------------------------------------------ public

    def draw(self, match: BoxingMatch, hud: HudInfo) -> None:
        self._surface.fill(BACKGROUND)
        self._draw_ring()
        for index, boxer in enumerate(match.boxers):
            self._draw_boxer(boxer, BOXER_COLORS[index])
        self._draw_hud(match, hud)
        self._present()

    def show_message(self, lines: list[str]) -> None:
        """Overlay centred text on top of the last drawn frame."""
        shade = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        shade.fill(MESSAGE_SHADE)
        self._surface.blit(shade, (0, 0))
        line_spacing = MESSAGE_FONT_SIZE + 8
        top = (WINDOW_HEIGHT - line_spacing * len(lines)) // 2
        for row, line in enumerate(lines):
            text = self._message_font.render(line, True, TEXT)
            x = (WINDOW_WIDTH - text.get_width()) // 2
            self._surface.blit(text, (x, top + row * line_spacing))
        self._present()

    def frame_array(self) -> np.ndarray:
        """The current frame as an RGB array of shape (height, width, 3)."""
        return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))

    def quit_requested(self) -> bool:
        """Process window events; True once the user closes the window or hits Esc."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def wait_for_next_frame(self, frames_per_second: float) -> None:
        """Sleep so that the game runs in real time."""
        self._clock.tick(frames_per_second)

    def close(self) -> None:
        pygame.quit()

    # ----------------------------------------------------------------- drawing

    def _present(self) -> None:
        if self._show_window:
            pygame.display.flip()

    def _to_screen(self, world_xy: np.ndarray) -> tuple[int, int]:
        center_x = WINDOW_WIDTH / 2
        center_y = HUD_HEIGHT + RING_MARGIN + RING_PIXELS / 2
        # World +Y is up on screen; screen Y grows downward.
        return (
            round(center_x + world_xy[0] * self._pixels_per_metre),
            round(center_y - world_xy[1] * self._pixels_per_metre),
        )

    def _to_pixels(self, metres: float) -> int:
        return max(1, round(metres * self._pixels_per_metre))

    def _draw_ring(self) -> None:
        ring_rect = pygame.Rect(
            RING_MARGIN, HUD_HEIGHT + RING_MARGIN, RING_PIXELS, RING_PIXELS
        )
        pygame.draw.rect(self._surface, CANVAS, ring_rect)
        pygame.draw.rect(self._surface, ROPE, ring_rect, ROPE_THICKNESS)
        pygame.draw.circle(
            self._surface,
            CENTER_MARK,
            self._to_screen(np.zeros(2)),
            self._to_pixels(self._config.rewards.center_control_radius),
            OUTLINE_THICKNESS,
        )

    def _draw_boxer(self, boxer: Boxer, color: tuple[int, int, int]) -> None:
        body = self._config.body
        body_color = EXHAUSTED_TINT if boxer.is_exhausted else color
        center = self._to_screen(boxer.position)
        pygame.draw.circle(self._surface, body_color, center, self._to_pixels(body.body_radius))

        if boxer.guard in (GuardState.HIGH, GuardState.LOW):
            pygame.draw.circle(
                self._surface,
                GUARD_OUTLINE,
                center,
                self._to_pixels(body.body_radius),
                OUTLINE_THICKNESS,
            )

        nose = boxer.position + heading_vector(boxer.facing) * body.body_radius
        pygame.draw.line(
            self._surface, BACKGROUND, center, self._to_screen(nose), FACING_LINE_THICKNESS
        )

        # A ducking head is drawn smaller: it is further from the camera.
        head_scale = boxer.head_height / body.standing_head_height
        pygame.draw.circle(
            self._surface, HEAD_COLOR, center, self._to_pixels(body.head_radius * head_scale)
        )

        for hand in Hand:
            pygame.draw.circle(
                self._surface,
                self._glove_color(boxer, hand),
                self._to_screen(boxer.gloves[hand].position[:2]),
                self._to_pixels(body.glove_radius),
            )

    @staticmethod
    def _glove_color(boxer: Boxer, hand: Hand) -> tuple[int, int, int]:
        if boxer.is_striking and boxer.strike_spec.hand is hand:
            return GLOVE_PHASE_COLORS[boxer.strike_phase]
        return GLOVE_REST

    def _draw_hud(self, match: BoxingMatch, hud: HudInfo) -> None:
        columns = (HUD_PADDING, WINDOW_WIDTH - HUD_PADDING - BAR_WIDTH)
        for index, boxer in enumerate(match.boxers):
            left = columns[index]
            self._draw_text(BOXER_LABELS[index], left, HUD_PADDING, BOXER_COLORS[index])
            self._draw_bar(left, HUD_PADDING + LINE_HEIGHT, boxer.health, HEALTH_COLOR)
            self._draw_bar(left, HUD_PADDING + 2 * LINE_HEIGHT, boxer.stamina, STAMINA_COLOR)
            self._draw_text(self._describe(boxer), left, HUD_PADDING + 3 * LINE_HEIGHT, TEXT)

        seconds_left = (
            match.time_remaining_fraction * self._config.timing.round_duration_seconds
        )
        footer = (
            f"{hud.status_text}   time {seconds_left:5.1f}s   "
            f"reward {hud.last_reward:+.2f} (round {hud.episode_reward:+.1f})"
        )
        self._draw_text(footer, HUD_PADDING, HUD_PADDING + 5 * LINE_HEIGHT, TEXT)

    @staticmethod
    def _describe(boxer: Boxer) -> str:
        if boxer.is_knocked_out:
            return "KNOCKED OUT"
        if boxer.is_exhausted:
            return "EXHAUSTED"
        if boxer.is_striking:
            return f"{boxer.strike_type.name} ({boxer.strike_phase.name.lower()})"
        return GUARD_DESCRIPTIONS[boxer.guard]

    def _draw_bar(self, left: int, top: int, fraction: float, color) -> None:
        pygame.draw.rect(self._surface, BAR_BACKGROUND, (left, top, BAR_WIDTH, BAR_HEIGHT))
        pygame.draw.rect(
            self._surface, color, (left, top, round(BAR_WIDTH * fraction), BAR_HEIGHT)
        )

    def _draw_text(self, text: str, left: int, top: int, color) -> None:
        self._surface.blit(self._font.render(text, True, color), (left, top))


# ------------------------------------------------------------------- keyboard

MOVE_KEYS = {
    pygame.K_w: np.array([0.0, 1.0]),
    pygame.K_s: np.array([0.0, -1.0]),
    pygame.K_a: np.array([-1.0, 0.0]),
    pygame.K_d: np.array([1.0, 0.0]),
}
STRIKE_KEYS = {
    pygame.K_j: StrikeType.JAB,
    pygame.K_k: StrikeType.CROSS,
    pygame.K_l: StrikeType.HOOK,
    pygame.K_i: StrikeType.UPPERCUT,
}
DEFENSE_KEYS = {
    pygame.K_q: GuardState.HIGH,
    pygame.K_e: GuardState.LOW,
    pygame.K_SPACE: GuardState.SLIP,
}
# A screen direction counts toward forward/strafe when it is within 67.5
# degrees of that axis (sin 22.5 deg), which splits the circle into 8 sectors.
DIRECTION_THRESHOLD = 0.38


class KeyboardController:
    """Human player. Requires an open Pygame window (render_mode="human").

    Movement is screen-relative (W is always up on screen), which is the
    natural mapping for a top-down view; it is converted to the boxer's own
    forward/strafe axes here.
    """

    def reset(self) -> None:
        return None

    def decide(self, match: BoxingMatch, boxer_index: int) -> BoxerCommand:
        pressed = pygame.key.get_pressed()
        boxer = match.boxers[boxer_index]

        screen_direction = sum(
            (direction for key, direction in MOVE_KEYS.items() if pressed[key]),
            start=np.zeros(2),
        )
        forward, strafe = self._to_local_steps(screen_direction, boxer.facing)
        return BoxerCommand(
            forward=forward,
            strafe=strafe,
            defense=self._first_pressed(pressed, DEFENSE_KEYS, GuardState.NONE),
            strike=self._first_pressed(pressed, STRIKE_KEYS, None),
        )

    @staticmethod
    def _to_local_steps(screen_direction: np.ndarray, facing: float) -> tuple[int, int]:
        length = float(np.linalg.norm(screen_direction))
        if length == 0.0:
            return 0, 0
        local = world_to_local(screen_direction / length, facing)
        steps = [
            int(np.sign(component)) if abs(component) > DIRECTION_THRESHOLD else 0
            for component in local
        ]
        return steps[0], steps[1]

    @staticmethod
    def _first_pressed(pressed, bindings: dict, default):
        for key, value in bindings.items():
            if pressed[key]:
                return value
        return default
