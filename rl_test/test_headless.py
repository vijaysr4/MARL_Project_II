import os

# Force Pygame to use a virtual/dummy buffer (No SSH Display needed)
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import cv2
import numpy as np
from pyquaticus.envs.pyquaticus import pyquaticus_v0


def test_render():
    print("Initializing environment in headless mode...")

    # We use rgb_array to tell the env to return pixel data
    try:
        env = pyquaticus_v0.PyQuaticusEnv(render_mode='rgb_array', team_size=3)
        env.reset()

        # Take one step to ensure the engine is drawing
        frame = env.render()

        if frame is not None:
            # Pygame is RGB, OpenCV is BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite("headless_test_frame.png", frame_bgr)
            print("SUCCESS: 'headless_test_frame.png' saved.")
        else:
            print("ERROR: Render returned None.")

        env.close()
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    test_render()