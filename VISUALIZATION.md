# Wrist motion visualization

For animated 3D orientation playback, run:

```bash
.venv/bin/python visualize_quaternion_orientation.py
# Or choose a raw or preprocessed recording:
.venv/bin/python visualize_quaternion_orientation.py path/to/part_001.csv --speed 0.5
```

The default is the first processed motion in `preprocessing/data`. The window
shows a rotating rectangular body and colored body axes (X red, Y green, Z blue).
Faint dashed axes mark the fixed reference frame. Use Play/Pause, Previous/Next,
Restart, or the sample slider to inspect the orientation and quaternion values.
Playback follows `secondsElapsed` at the requested speed and stops at the end.
On a slow display, playback can skip samples to maintain timing; Previous/Next
lets you inspect every sample individually. Drag the 3D view to change the camera.

Requires NumPy and Matplotlib from `requirements.txt` and an interactive desktop
Matplotlib backend. Quaternions are normalized and interpreted as scalar-first
Hamilton active rotations from body coordinates into reference coordinates.
Processed recordings therefore start aligned with the reference axes. Raw
recordings retain their supplied reference frame. This displays orientation only;
it does not infer position or a complete arm pose.

Generate the self-contained dashboard with the first recording in the dataset:

```bash
python3 visualize_wrist_motion.py
```

Or embed a specific recording:

```bash
python3 visualize_wrist_motion.py Whales1and2_Raw_Labelled/<recording>.csv -o wrist_motion_dashboard.html
```

Open `wrist_motion_dashboard.html` in a browser. The dashboard groups the sensor
columns into rotation rate, gravity, acceleration, and quaternion panels, all
against `secondsElapsed`. It displays `activity`, `activityEncoded`, `weight`,
`set`, and `reps` as recording context. Use **Open another CSV** to switch to
another compatible recording without regenerating the file.

The generator uses only the Python standard library. Recordings above 4,000
observations are evenly downsampled for responsive display; missing values are
shown as gaps rather than zeros. Change the limit with `--max-points`.
