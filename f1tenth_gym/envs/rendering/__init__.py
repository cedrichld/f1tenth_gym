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
    "F1TENTH rendering needs an X display (real or virtual), but no $DISPLAY was found.\n"
    "  • Headless server: run your script under a virtual display:\n"
    "        xvfb-run -a python your_script.py\n"
    "  • Google Colab (in a cell, before creating the env):\n"
    "        !apt-get -qq install -y xvfb\n"
    "        !pip -q install pyvirtualdisplay\n"
    "        from pyvirtualdisplay import Display\n"
    "        Display(visible=0, size=(800, 800)).start()\n"
    "    then use render_mode='rgb_array' and embed the frames/video inline."
)

# Render modes that draw pixels and therefore require a display. (gymnasium
# strips the "_list" suffix, so the env only ever sees "rgb_array" here.)
_DISPLAY_RENDER_MODES = ("human", "human_fast", "unlimited", "rgb_array")


def make_renderer(
    params: VehicleParameters,
    track: "Track",
    agent_ids: list[str],
    render_mode: Optional[str] = None,
    render_fps: Optional[int] = 60,
    render_spec: RenderSpec = RenderSpec(),
) -> tuple[EnvRenderer, RenderSpec]:
    """Return a GL renderer and the render spec.

    All rendering uses the OpenGL backend (``PyQtEnvRendererGL``), which needs
    an X display -- real or virtual (``xvfb``). If a display render mode is
    requested with no ``$DISPLAY``, raise a clear error with setup guidance
    rather than failing cryptically deep in Qt.
    """
    needs_display = render_mode in _DISPLAY_RENDER_MODES
    if needs_display and not os.environ.get("DISPLAY"):
        raise RuntimeError(NO_DISPLAY_GUIDANCE)

    if render_mode == "rgb_array":
        # off-screen grab from the GL framebuffer under a real/virtual X server
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    from .rendering_pyqtgl import PyQtEnvRendererGL as EnvRenderer

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
