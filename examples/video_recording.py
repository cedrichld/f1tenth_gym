"""Record an rgb_array rollout to an MP4 with gymnasium's RecordVideo wrapper.

rgb_array rendering uses the GL backend by default (fast framebuffer grab),
which needs an X display -- real or virtual. On a desktop with a display this
just works. On a headless server or in Google Colab, start a virtual display
first:

    # shell / headless server:
    xvfb-run -a python examples/video_recording.py

    # Google Colab (in a cell, before creating the env):
    #   !apt-get -qq install -y xvfb
    #   !pip -q install pyvirtualdisplay
    #   from pyvirtualdisplay import Display
    #   Display(visible=0, size=(800, 800)).start()
    # then run this script's main(); embed the result inline with:
    #   from IPython.display import Video
    #   Video("video_<timestamp>/rl-video-episode-0.mp4", embed=True)

To skip the display requirement and use the slower pure-CPU renderer, pass
RenderConfig(frame_output_method="2d") in the EnvConfig below.
"""
import time
import gymnasium as gym
import gymnasium.wrappers
import numpy as np

from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.reset import ResetStrategy
from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig, ResetConfig
from waypoint_follow import PurePursuitPlanner

def main():
    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.82461887897713965,
        "vgain": 1,
    }

    cfg = EnvConfig(
        observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
        reset_config=ResetConfig(strategy=ResetStrategy.RL_RANDOM_STATIC),
    )

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=cfg,
        render_mode="rgb_array",
    )
    env = gymnasium.wrappers.RecordVideo(env, f"video_{time.time()}")
    track = env.unwrapped.track

    planner = PurePursuitPlanner(track=track, wb=0.17145 + 0.15875)
    track.raceline.render_waypoints(env.unwrapped.renderer)
    for r in planner.get_render_callbacks():
        env.unwrapped.add_render_callback(r)

    poses = np.array(
        [
            [
                track.raceline.xs[0],
                track.raceline.ys[0],
                track.raceline.yaws[0],
            ]
        ]
    )

    obs, info = env.reset(options={"poses": poses})
    done = False

    laptime = 0.0
    start = time.time()

    frames = [env.render()]
    while not done and laptime < 5.0:
        action = env.action_space.sample()
        for i, agent_id in enumerate(obs.keys()):
            speed, steer = planner.plan(
                obs[agent_id]["pose_x"],
                obs[agent_id]["pose_y"],
                obs[agent_id]["pose_theta"],
                work["tlad"],
                work["vgain"],
            )
            action[i] = [steer, speed]

        obs, step_reward, done, truncated, info = env.step(action)
        laptime += step_reward

        frame = env.render()
        frames.append(frame)

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)

    # close env to trigger video saving
    env.close()


if __name__ == "__main__":
    main()
