# Results Display

Centralized visual outputs for the baseline models. Each baseline writes its
rendered images, GIFs, and videos below its own directory.

Set `ANYSOLE_RESULTSDISPLAY` to override this root.

Visualization scripts:

- `script/visualize_motionpro.py`: MotionPRO tactile/prediction/GT comparison.
- `script/visualize_step2motion.py`: Step2Motion tactile/generated-BVH/GT comparison.

Run from the repository root with `python resultsdisplay/script/<script>.py`.

'''python
conda activate touch_gait

python resultsdisplay/script/visualize_motionpro.py

python resultsdisplay/script/visualize_step2motion.py
'''