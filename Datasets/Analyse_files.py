import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Force Plotly to open in a web browser
pio.renderers.default = "browser"

# Load the npy file
npy_file_path = "results/2025-02-28_22-23-37/livox_horizon_poses.npy"
trajectory_data = np.load(npy_file_path)  # Shape: (N, 4, 4)

# Extract positions (translation part of the 4x4 transformation matrices)
positions = trajectory_data[:, :3, 3]  # Extract (x, y, z) from each transformation matrix

# Compute total distance traveled (sum of Euclidean distances between consecutive points)
distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
total_distance = np.sum(distances)

# Compute total displacement (Euclidean distance between first and last points)
total_displacement = np.linalg.norm(positions[-1] - positions[0])

# Create an interactive 3D trajectory plot
fig = go.Figure()

# Add trajectory
fig.add_trace(go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    mode='lines',
    line=dict(color='blue', width=2),
    name="LiDAR Trajectory"
))

# Add point numbers every 10th point
for i in range(0, len(positions), 10):
    fig.add_trace(go.Scatter3d(
        x=[positions[i, 0]],
        y=[positions[i, 1]],
        z=[positions[i, 2]],
        mode='text',
        text=str(i+1),  # Label each point with its index (starting from 1)
        textposition="top center",
        showlegend=False
    ))

# Set plot layout
fig.update_layout(
    title=f"Interactive 3D LiDAR Trajectory<br>Total Distance: {total_distance:.2f}m | Displacement: {total_displacement:.2f}m",
    scene=dict(
        xaxis_title="X Position",
        yaxis_title="Y Position",
        zaxis_title="Z Position"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Show the plot in a web browser
fig.show(config={"scrollZoom": True})

# Print relevant metrics
print(f"Total Distance Traveled: {total_distance:.2f} meters")
print(f"Total Displacement: {total_displacement:.2f} meters")
