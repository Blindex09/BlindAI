"""Guia de enquadramento facial, filtro contra tremores, histerese, estabilidade e selecao de rajada."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .face_engine import FaceDetection

STABILITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "normal": {
        "min_cutoff": 1.2,
        "beta": 0.05,
        "cx_in": 0.085,
        "cx_out": 0.130,
        "cy_in": (0.35, 0.58),
        "cy_out": (0.30, 0.63),
        "size_in": (0.24, 0.56),
        "size_out": (0.20, 0.62),
        "roll_in": 12.0,
        "roll_out": 17.0,
        "yaw_in": 0.16,
        "yaw_out": 0.22,
        "mad_xy": 0.018,
        "vel_xy": 0.065,
        "grace_s": 0.55,
    },
    "gentle": {
        "min_cutoff": 0.7,
        "beta": 0.03,
        "cx_in": 0.105,
        "cx_out": 0.155,
        "cy_in": (0.33, 0.61),
        "cy_out": (0.28, 0.66),
        "size_in": (0.21, 0.60),
        "size_out": (0.17, 0.66),
        "roll_in": 16.0,
        "roll_out": 22.0,
        "yaw_in": 0.20,
        "yaw_out": 0.26,
        "mad_xy": 0.032,
        "vel_xy": 0.115,
        "grace_s": 0.85,
    },
    "strict": {
        "min_cutoff": 1.8,
        "beta": 0.08,
        "cx_in": 0.065,
        "cx_out": 0.100,
        "cy_in": (0.38, 0.54),
        "cy_out": (0.34, 0.58),
        "size_in": (0.26, 0.52),
        "size_out": (0.22, 0.57),
        "roll_in": 8.0,
        "roll_out": 12.0,
        "yaw_in": 0.12,
        "yaw_out": 0.17,
        "mad_xy": 0.011,
        "vel_xy": 0.040,
        "grace_s": 0.35,
    },
}


def resolve_stability_mode(mode: str) -> str:
    m = (mode or "normal").strip().lower()
    if m in ("suave", "gentle", "tremor", "tolerante"):
        return "gentle"
    if m in ("preciso", "strict", "rigoroso", "exato"):
        return "strict"
    if m in ("normal", "padrao", "padrão", "default"):
        return "normal"
    raise ValueError("Modo de estabilidade inválido. Escolha entre normal, gentle ou strict.")


class OneEuroFilter:
    """Filtro 1-Euro adaptativo: suaviza tremores parado e acompanha movimentos rapidos sem atraso."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: Optional[float] = None
        self._dx: float = 0.0
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = None

    def update(self, value: float, t: float) -> float:
        val = float(value)
        ts = float(t)
        if self._x is None or self._t is None:
            self._x = val
            self._dx = 0.0
            self._t = ts
            return val
        dt = max(ts - self._t, 1e-4)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (val - self._x) / dt
        edx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(cutoff, dt)
        x_hat = a * val + (1.0 - a) * self._x
        self._x = x_hat
        self._dx = edx
        self._t = ts
        return x_hat


class FramingEvaluator:
    """Avalia enquadramento facial com suavizacao 1-Euro, histerese e uma instrucao clara por vez."""

    def __init__(self, stability_mode: str = "normal", mirror_horizontal: bool = False) -> None:
        self.stability_mode = resolve_stability_mode(stability_mode)
        self.profile = STABILITY_PROFILES[self.stability_mode]
        self.mirror_horizontal = bool(mirror_horizontal)
        mc = self.profile["min_cutoff"]
        bt = self.profile["beta"]
        self._f_cx = OneEuroFilter(mc, bt)
        self._f_cy = OneEuroFilter(mc, bt)
        self._f_h = OneEuroFilter(mc, bt)
        self._f_w = OneEuroFilter(mc, bt)
        self._f_roll = OneEuroFilter(mc, bt)
        self._f_yaw = OneEuroFilter(mc, bt)
        self._axis_ok: Dict[str, bool] = {}
        self._lighting_warned: set[str] = set()

    def reset_filters(self) -> None:
        for flt in (self._f_cx, self._f_cy, self._f_h, self._f_w, self._f_roll, self._f_yaw):
            flt.reset()
        self._axis_ok.clear()

    def _hyst(self, axis: str, inside_in: bool, inside_out: bool) -> bool:
        was_ok = self._axis_ok.get(axis, False)
        now_ok = bool(inside_out if was_ok else inside_in)
        self._axis_ok[axis] = now_ok
        return now_ok

    def evaluate(
        self,
        det: Optional[FaceDetection],
        t: float,
        quality: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Devolve diagnostico completo e uma unica instrucao prioritaria em portugues."""
        lighting_status = (quality or {}).get("lighting_status", "ok")
        lighting_tip = ""
        severe_lighting = lighting_status in ("too_dark", "backlit", "too_bright")
        if lighting_status == "too_dark":
            lighting_tip = "O ambiente está escuro; se puder, acenda uma luz ou vire para uma janela."
        elif lighting_status == "backlit":
            lighting_tip = "Há muita luz forte atrás de você; vire de frente para a luz ou feche a cortina."
        elif lighting_status == "too_bright":
            lighting_tip = "Há luz forte demais no rosto; afaste um pouco de luz direta."
        elif lighting_status == "side_shadow":
            lighting_tip = "Um lado do rosto está na sombra; se quiser, vire um pouco na direção da luz."

        if det is None:
            self.reset_filters()
            spoken = "Não estou vendo o seu rosto. Fique na frente da câmera e olhe na direção da tela."
            if lighting_status == "too_dark":
                spoken = "Não estou vendo o seu rosto e o ambiente está bem escuro. Acenda uma luz e fique na frente da câmera."
            return {
                "face_detected": False,
                "status": "no_face",
                "issue_code": "no_face",
                "is_centered": False,
                "is_in_outer_zone": False,
                "framing_score": 0.0,
                "spoken_instruction": spoken,
                "lighting_tip": lighting_tip,
                "filtered": None,
            }

        cx_f = self._f_cx.update(det.cx, t)
        cy_f = self._f_cy.update(det.cy, t)
        h_f = self._f_h.update(det.h, t)
        w_f = self._f_w.update(det.w, t)
        roll_f = self._f_roll.update(det.roll_deg, t)
        yaw_f = self._f_yaw.update(det.yaw_ratio, t)

        x_f = cx_f - w_f / 2.0
        y_f = cy_f - h_f / 2.0
        head_top_f = y_f - 0.20 * h_f
        chin_bottom_f = y_f + h_f
        right_edge_f = x_f + w_f

        p = self.profile
        cut_top_ok = self._hyst("cut_top", head_top_f >= 0.020, head_top_f >= -0.015)
        cut_bottom_ok = self._hyst("cut_bottom", chin_bottom_f <= 0.960, chin_bottom_f <= 0.995)
        cut_left_ok = self._hyst("cut_left", x_f >= 0.020, x_f >= -0.015)
        cut_right_ok = self._hyst("cut_right", right_edge_f <= 0.980, right_edge_f <= 1.015)

        dist_in = p["size_in"][0] <= h_f <= p["size_in"][1]
        dist_out = p["size_out"][0] <= h_f <= p["size_out"][1]
        dist_ok = self._hyst("distance", dist_in, dist_out)

        hx_err = abs(cx_f - 0.5)
        horiz_ok = self._hyst("horiz", hx_err <= p["cx_in"], hx_err <= p["cx_out"])

        vert_in = p["cy_in"][0] <= cy_f <= p["cy_in"][1]
        vert_out = p["cy_out"][0] <= cy_f <= p["cy_out"][1]
        vert_ok = self._hyst("vert", vert_in, vert_out)

        yaw_err = abs(yaw_f - 0.5)
        yaw_ok = self._hyst("yaw", yaw_err <= p["yaw_in"], yaw_err <= p["yaw_out"])
        roll_ok = self._hyst("roll", abs(roll_f) <= p["roll_in"], abs(roll_f) <= p["roll_out"])

        margins_out_ok = (
            head_top_f >= -0.015 and chin_bottom_f <= 0.995 and x_f >= -0.015 and right_edge_f <= 1.015
        )
        is_in_outer_zone = bool(margins_out_ok and dist_out and hx_err <= p["cx_out"] and vert_out)

        issue_code = "centered"
        instruction = "Rosto bem centralizado."

        # Seleciona apenas a correcao mais importante por vez para nao confundir na fala.
        if not cut_top_ok:
            issue_code = "cut_top"
            instruction = "O alto da cabeça está cortado. Abaixe um pouco a cabeça ou afaste um pouco da câmera."
        elif not cut_bottom_ok:
            issue_code = "cut_bottom"
            instruction = "O queixo está cortado embaixo. Levante um pouco a cabeça ou incline a câmera para baixo."
        elif not cut_left_ok:
            issue_code = "cut_left"
            side = "direita" if self.mirror_horizontal else "esquerda"
            instruction = f"Seu rosto está cortado na borda. Mova para a sua {side}."
        elif not cut_right_ok:
            issue_code = "cut_right"
            side = "esquerda" if self.mirror_horizontal else "direita"
            instruction = f"Seu rosto está cortado na borda. Mova para a sua {side}."
        elif not dist_ok:
            if h_f > p["size_in"][1]:
                issue_code = "too_close"
                instruction = "Você está muito perto da câmera. Afaste um pouco para trás."
            else:
                issue_code = "too_far"
                instruction = "Você está um pouco longe da câmera. Chegue mais perto."
        elif not horiz_ok:
            intensity = "mais" if hx_err >= 0.18 else "um pouco"
            if cx_f < 0.5:
                # Padrao DirectShow (mirror_horizontal=False): rosto na esquerda da camera -> pessoa esta deslocada
                # para a propria esquerda/direita conforme convencao configurada.
                move_word = "direita" if self.mirror_horizontal else "esquerda"
                issue_code = f"move_{'right' if self.mirror_horizontal else 'left'}"
                instruction = f"Mova {intensity} para a sua {move_word}."
            else:
                move_word = "esquerda" if self.mirror_horizontal else "direita"
                issue_code = f"move_{'left' if self.mirror_horizontal else 'right'}"
                instruction = f"Mova {intensity} para a sua {move_word}."
        elif not vert_ok:
            if cy_f < p["cy_in"][0]:
                issue_code = "move_down"
                instruction = "Você está um pouco alto na imagem. Abaixe um pouco ou incline a tela para cima."
            else:
                issue_code = "move_up"
                instruction = "Você está um pouco baixo na imagem. Levante um pouco a cabeça ou incline a tela para baixo."
        elif not yaw_ok:
            issue_code = "turn_face"
            if yaw_f > 0.5:
                turn_word = "esquerda" if self.mirror_horizontal else "direita"
            else:
                turn_word = "direita" if self.mirror_horizontal else "esquerda"
            instruction = f"Vire o rosto um pouco para a sua {turn_word} para olhar para a câmera."
        elif not roll_ok:
            issue_code = "tilt_head"
            tilt_dir = "direita" if ((roll_f < 0) ^ self.mirror_horizontal) else "esquerda"
            instruction = f"A cabeça está um pouco inclinada. Endireite inclinando leve para a sua {tilt_dir}."

        # Avisa iluminacao ruim uma vez com clareza, sem travar o usuario se ele nao puder mudar a luz.
        if severe_lighting and lighting_status not in self._lighting_warned:
            self._lighting_warned.add(lighting_status)
            if issue_code == "centered":
                instruction = f"Rosto centralizado, mas {lighting_tip[0].lower() + lighting_tip[1:]}"
            else:
                instruction = f"{instruction} {lighting_tip}"

        is_centered = issue_code == "centered"
        # Score continuo de enquadramento (0.0 a 1.0) para desempate na selecao de melhor quadro.
        target_cy = (p["cy_in"][0] + p["cy_in"][1]) / 2.0
        target_size = (p["size_in"][0] + p["size_in"][1]) / 2.0
        score = 1.0
        score -= min(0.45, hx_err * 1.4)
        score -= min(0.25, abs(cy_f - target_cy) * 1.2)
        score -= min(0.15, abs(h_f - target_size) * 0.8)
        score -= min(0.10, yaw_err * 0.4)
        score -= min(0.05, abs(roll_f) / 90.0)
        if not margins_out_ok:
            score -= 0.25
        framing_score = max(0.0, min(1.0, score))

        return {
            "face_detected": True,
            "status": "centered" if is_centered else "adjust",
            "issue_code": issue_code,
            "is_centered": is_centered,
            "is_in_outer_zone": is_in_outer_zone,
            "framing_score": round(framing_score, 4),
            "spoken_instruction": instruction,
            "lighting_tip": lighting_tip,
            "filtered": {
                "cx": round(cx_f, 4),
                "cy": round(cy_f, 4),
                "w": round(w_f, 4),
                "h": round(h_f, 4),
                "head_top_y": round(head_top_f, 4),
                "chin_bottom_y": round(chin_bottom_f, 4),
                "roll_deg": round(roll_f, 2),
                "yaw_ratio": round(yaw_f, 3),
            },
        }


class StillnessTracker:
    """Detecta quando o rosto esta parado de forma tolerante a pequenos tremores involuntarios."""

    def __init__(
        self,
        stability_mode: str = "normal",
        hold_seconds: float = 2.0,
        window_seconds: float = 0.9,
    ) -> None:
        self.stability_mode = resolve_stability_mode(stability_mode)
        self.profile = STABILITY_PROFILES[self.stability_mode]
        self.hold_seconds = max(0.5, float(hold_seconds))
        self.window_seconds = max(0.35, float(window_seconds))
        self._history: Deque[Tuple[float, float, float, float, float]] = deque()
        self._hold_elapsed: float = 0.0
        self._last_t: Optional[float] = None
        self._grace_started_at: Optional[float] = None
        self._last_countdown_num: Optional[int] = None

    def _compute_motion(self) -> Tuple[float, float, bool]:
        if len(self._history) < 2:
            return 0.0, 0.0, False
        ts = np.array([row[0] for row in self._history], dtype=np.float64)
        span = float(ts[-1] - ts[0])
        xs_raw = np.array([row[1] for row in self._history], dtype=np.float64)
        ys_raw = np.array([row[2] for row in self._history], dtype=np.float64)

        # MAD (Median Absolute Deviation) ignora espasmos curtos de 1-2 quadros enquanto mede a dispersao real.
        med_x = float(np.median(xs_raw))
        med_y = float(np.median(ys_raw))
        mad_x = float(np.median(np.abs(xs_raw - med_x)))
        mad_y = float(np.median(np.abs(ys_raw - med_y)))
        mad_xy = math.hypot(mad_x, mad_y)

        # Velocidade filtrada entre os dois ultimos quadros.
        dt = max(float(self._history[-1][0] - self._history[-2][0]), 1e-3)
        dx_f = float(self._history[-1][3] - self._history[-2][3])
        dy_f = float(self._history[-1][4] - self._history[-2][4])
        vel_xy = math.hypot(dx_f, dy_f) / dt

        ready = len(self._history) >= 4 and span >= 0.35
        return mad_xy, vel_xy, ready

    def update(
        self,
        det: Optional[FaceDetection],
        framing: Dict[str, Any],
        t: float,
    ) -> Dict[str, Any]:
        ts = float(t)
        while self._history and (ts - self._history[0][0]) > self.window_seconds:
            self._history.popleft()

        if det is None or not framing.get("face_detected"):
            had_hold = self._hold_elapsed > 0.0
            self._history.clear()
            self._hold_elapsed = 0.0
            self._last_t = ts
            self._grace_started_at = None
            self._last_countdown_num = None
            return {
                "state": "SEARCHING",
                "spoken_cue": "Perdi o rosto, volte devagar para a frente da câmera." if had_hold else framing["spoken_instruction"],
                "sound_cue": "warn" if had_hold else "",
                "hold_elapsed": 0.0,
                "hold_remaining": self.hold_seconds,
                "countdown_number": None,
                "mad_xy": 0.0,
                "velocity_xy": 0.0,
                "in_grace": False,
            }

        flt = framing["filtered"]
        self._history.append((ts, det.cx, det.cy, flt["cx"], flt["cy"]))
        mad_xy, vel_xy, window_ready = self._compute_motion()

        is_steady_kinematics = (
            window_ready
            and mad_xy <= self.profile["mad_xy"]
            and vel_xy <= self.profile["vel_xy"]
        )
        dt = 0.0 if self._last_t is None else max(0.0, min(0.5, ts - self._last_t))
        self._last_t = ts

        is_centered = bool(framing.get("is_centered"))
        is_in_outer = bool(framing.get("is_in_outer_zone"))

        # Ja iniciou a contagem de parado: protege com buffer de tolerancia para movimentos involuntarios breves.
        if self._hold_elapsed > 0.0 or self._grace_started_at is not None:
            if is_in_outer and is_steady_kinematics:
                self._grace_started_at = None
                self._hold_elapsed += dt
                if self._hold_elapsed >= self.hold_seconds:
                    return {
                        "state": "CAPTURE_NOW",
                        "spoken_cue": "Tirando a foto!",
                        "sound_cue": "shutter",
                        "hold_elapsed": round(self._hold_elapsed, 3),
                        "hold_remaining": 0.0,
                        "countdown_number": 0,
                        "mad_xy": round(mad_xy, 4),
                        "velocity_xy": round(vel_xy, 4),
                        "in_grace": False,
                    }
                rem = max(1, math.ceil(self.hold_seconds - self._hold_elapsed))
                cue = ""
                sound = ""
                if rem != self._last_countdown_num:
                    self._last_countdown_num = rem
                    palavras = {1: "Um.", 2: "Dois.", 3: "Três.", 4: "Quatro.", 5: "Cinco."}
                    cue = palavras.get(rem, f"{rem}.")
                    sound = "tick"
                return {
                    "state": "COUNTDOWN",
                    "spoken_cue": cue,
                    "sound_cue": sound,
                    "hold_elapsed": round(self._hold_elapsed, 3),
                    "hold_remaining": round(max(0.0, self.hold_seconds - self._hold_elapsed), 3),
                    "countdown_number": rem,
                    "mad_xy": round(mad_xy, 4),
                    "velocity_xy": round(vel_xy, 4),
                    "in_grace": False,
                }

            # Espasmo breve ainda dentro da zona externa: pausa o relogio sem resetar nem xingar direcao.
            if is_in_outer:
                if self._grace_started_at is None:
                    self._grace_started_at = ts
                if (ts - self._grace_started_at) <= self.profile["grace_s"]:
                    return {
                        "state": "STEADY_HOLD",
                        "spoken_cue": "",
                        "sound_cue": "",
                        "hold_elapsed": round(self._hold_elapsed, 3),
                        "hold_remaining": round(max(0.0, self.hold_seconds - self._hold_elapsed), 3),
                        "countdown_number": None,
                        "mad_xy": round(mad_xy, 4),
                        "velocity_xy": round(vel_xy, 4),
                        "in_grace": True,
                    }

            # Saiu da zona externa ou se mexeu por mais tempo que a janela de tolerancia.
            self._hold_elapsed = 0.0
            self._grace_started_at = None
            self._last_countdown_num = None
            cue = (
                "Você se mexeu um pouco, vamos ajustar de novo."
                if is_centered
                else f"Você se mexeu um pouco. {framing['spoken_instruction']}"
            )
            return {
                "state": "GUIDING",
                "spoken_cue": cue,
                "sound_cue": "warn",
                "hold_elapsed": 0.0,
                "hold_remaining": self.hold_seconds,
                "countdown_number": None,
                "mad_xy": round(mad_xy, 4),
                "velocity_xy": round(vel_xy, 4),
                "in_grace": False,
            }

        # Ainda nao comecou a contagem de parado.
        if not is_centered:
            return {
                "state": "GUIDING",
                "spoken_cue": framing["spoken_instruction"],
                "sound_cue": "",
                "hold_elapsed": 0.0,
                "hold_remaining": self.hold_seconds,
                "countdown_number": None,
                "mad_xy": round(mad_xy, 4),
                "velocity_xy": round(vel_xy, 4),
                "in_grace": False,
            }

        if not is_steady_kinematics:
            return {
                "state": "CENTERED_MOVING",
                "spoken_cue": "Rosto centralizado. Fique parado um instante.",
                "sound_cue": "centered",
                "hold_elapsed": 0.0,
                "hold_remaining": self.hold_seconds,
                "countdown_number": None,
                "mad_xy": round(mad_xy, 4),
                "velocity_xy": round(vel_xy, 4),
                "in_grace": False,
            }

        # Acabou de centralizar e estabilizar!
        self._hold_elapsed = max(dt, 0.02)
        self._last_countdown_num = None
        return {
            "state": "STEADY_HOLD",
            "spoken_cue": "Perfeito, está parado. Não se mexa.",
            "sound_cue": "centered",
            "hold_elapsed": round(self._hold_elapsed, 3),
            "hold_remaining": round(max(0.0, self.hold_seconds - self._hold_elapsed), 3),
            "countdown_number": math.ceil(max(0.0, self.hold_seconds - self._hold_elapsed)),
            "mad_xy": round(mad_xy, 4),
            "velocity_xy": round(vel_xy, 4),
            "in_grace": False,
        }


class ObjectStabilityTracker:
    """Espera estabilizacao da mao/objeto para capturar fotos de objetos, dinheiro ou textos."""

    def __init__(self, stability_mode: str = "normal", hold_seconds: float = 0.9) -> None:
        self.stability_mode = resolve_stability_mode(stability_mode)
        self.hold_seconds = max(0.35, float(hold_seconds))
        self._hold_elapsed: float = 0.0
        self._last_t: Optional[float] = None
        self._grace_started_at: Optional[float] = None
        diff_limits = {"normal": 4.5, "gentle": 7.5, "strict": 2.5}
        grace_limits = {"normal": 0.45, "gentle": 0.70, "strict": 0.25}
        self._max_diff = diff_limits[self.stability_mode]
        self._grace_s = grace_limits[self.stability_mode]

    def update(self, motion_diff: float, quality: Dict[str, Any], t: float) -> Dict[str, Any]:
        ts = float(t)
        dt = 0.0 if self._last_t is None else max(0.0, min(0.5, ts - self._last_t))
        self._last_t = ts
        lighting = quality.get("lighting_status", "ok")
        sharp_ok = quality.get("sharpness_status") != "blurry"
        steady = motion_diff <= self._max_diff

        if lighting == "too_dark":
            cue = "O ambiente está muito escuro. Se puder, acenda uma luz e mantenha firme."
        elif not steady:
            cue = "Segure firme um instante para não sair borrado."
        elif not sharp_ok:
            cue = "Aproxime ou afaste devagar para focar melhor e mantenha firme."
        else:
            cue = "Firme, mantendo o foco."

        if steady:
            self._grace_started_at = None
            self._hold_elapsed += dt
            if self._hold_elapsed >= self.hold_seconds:
                return {"state": "CAPTURE_NOW", "spoken_cue": "Tirando a foto!", "hold_elapsed": round(self._hold_elapsed, 3)}
            return {"state": "STEADY_HOLD", "spoken_cue": cue, "hold_elapsed": round(self._hold_elapsed, 3)}

        if self._hold_elapsed > 0.0:
            if self._grace_started_at is None:
                self._grace_started_at = ts
            if (ts - self._grace_started_at) <= self._grace_s:
                return {"state": "STEADY_HOLD", "spoken_cue": "", "hold_elapsed": round(self._hold_elapsed, 3)}
        self._hold_elapsed = 0.0
        self._grace_started_at = None
        return {"state": "GUIDING", "spoken_cue": cue, "hold_elapsed": 0.0}


@dataclass
class BurstFrame:
    """Um quadro candidato capturado em rajada para escolher o mais nitido e centralizado."""

    rgb: np.ndarray
    sharpness: float
    framing_score: float = 1.0
    pose_score: float = 1.0
    lighting_score: float = 1.0
    face_detected: bool = True


def select_best_burst_frame(candidates: List[BurstFrame]) -> Tuple[int, BurstFrame]:
    """Escolhe o melhor quadro da rajada combinando nitidez, centralizacao, olhar frontal e luz."""
    if not candidates:
        raise ValueError("Lista de quadros da rajada vazia.")
    if len(candidates) == 1:
        return 0, candidates[0]

    max_sharp = max(max(c.sharpness for c in candidates), 1e-3)
    med_sharp = float(np.median([c.sharpness for c in candidates]))

    best_idx = 0
    best_total = -1e9
    for idx, c in enumerate(candidates):
        norm_sharp = max(0.0, min(1.0, c.sharpness / max_sharp))
        # Penaliza quadros que perderam subitamente nitidez em relacao a mediana da rajada (piscada/tremida).
        blur_penalty = 0.25 if med_sharp > 10.0 and c.sharpness < 0.65 * med_sharp else 0.0
        face_bonus = 0.15 if c.face_detected else 0.0
        total = (
            0.45 * norm_sharp
            + 0.30 * float(c.framing_score)
            + 0.15 * float(c.pose_score)
            + 0.10 * float(c.lighting_score)
            + face_bonus
            - blur_penalty
        )
        if total > best_total:
            best_total = total
            best_idx = idx
    return best_idx, candidates[best_idx]
