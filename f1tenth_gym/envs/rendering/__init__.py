import os
from typing import Optional, TYPE_CHECKING

from .renderer import RenderSpec, EnvRenderer, ObjectRenderer
from .callbacks import make_lidar_scan_callback
from ..dynamic_models import VehicleParameters

if TYPE_CHECKING:
    from ..track import Track

__all__ = [
    "RenderSpec",
    "EnvRenderer",
    "ObjectRenderer",
    "make_lidar_scan_callback",
    "make_renderer",
]

NO_DISPLAY_GUIDANCE = (
    "rgb_array rendering with the GL backend needs an X display (real or virtual), "
    "but no $DISPLAY was found.\n"
    "  • Headless server / Google Colab: install xvfb and start a virtual display, e.g.\n"
    "        apt-get install -y xvfb            # in Colab, prefix with '!'\n"
    "        pip install pyvirtualdisplay       # in Colab, prefix with '!'\n"
    "        from pyvirtualdisplay import Display\n"
    "        Display(visible=0, size=(800, 800)).start()\n"
    "    or run your whole script under:  xvfb-run -a python your_script.py\n"
    "  • To use the slower pure-CPU renderer instead (no display needed), set\n"
    "        EnvConfig(render_config=RenderConfig(frame_output_method='2d'))\n"
    "  • To fall back to CPU automatically when no display is present, use\n"
    "        frame_output_method='auto'"
)


def _resolve_offscreen_backend(method: str) -> str:
    """Resolve the rgb_array backend to ``"gl"`` or ``"2d"``.

    The GL widget cannot render under the headless ``offscreen`` Qt platform
    (it has no FBO there), so a GL grab needs a real X server or a virtual one
    (``xvfb``), signalled by ``$DISPLAY``.

    * ``"gl"``   -> GL; **raises** ``RuntimeError`` with setup guidance if no display.
    * ``"2d"``   -> the 2D raster exporter (no display needed).
    * ``"auto"`` -> GL if ``$DISPLAY`` is set, else 2D (silent headless fallback).
    """
    has_display = bool(os.environ.get("DISPLAY"))
    if method == "2d":
        return "2d"
    if method == "auto":
        return "gl" if has_display else "2d"
    # method == "gl" (default): require a display, guide the user if missing.
    if not has_display:
        raise RuntimeError(NO_DISPLAY_GUIDANCE)
    return "gl"


def make_renderer(
    params: VehicleParameters,
    track: "Track",
    agent_ids: list[str],
    render_mode: Optional[str] = None,
    render_fps: Optional[int] = 60,
    render_spec: RenderSpec = RenderSpec(),
) -> tuple[EnvRenderer, RenderSpec]:
    """Return an instance of the renderer and the rendering specification."""
    is_rgb = render_mode in ("rgb_array", "rgb_array_list")
    method = getattr(render_spec, "frame_output_method", "auto")

    if render_spec.render_type == "pyqt6":
        if is_rgb:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from .rendering_pyqt import PyQtEnvRenderer as EnvRenderer
    elif render_spec.render_type == "pyqt6gl":
        if is_rgb:
            if _resolve_offscreen_backend(method) == "gl":
                # fast GL framebuffer grab; needs a display (real X or xvfb)
                os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
                from .rendering_pyqtgl import PyQtEnvRendererGL as EnvRenderer
            else:
                # 2D raster exporter (no display needed)
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
                from .rendering_pyqt import PyQtEnvRenderer as EnvRenderer
        else:
            from .rendering_pyqtgl import PyQtEnvRendererGL as EnvRenderer
    else:
        raise ValueError(f"Unknown render type: {render_spec.render_type}")

    if render_mode in ["human", "rgb_array", "unlimited", "human_fast"]:
        renderer = EnvRenderer(
            params=params,
            track=track,
            agent_ids=agent_ids,
            render_spec=render_spec,
            render_mode=render_mode,
            render_fps=render_fps,
        )
    else:
        renderer = None
    return renderer, render_spec
