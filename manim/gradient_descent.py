from manim import *
import numpy as np


class GradientDescent3D(ThreeDScene):
    def construct(self):
        # --- 1. Setup the Scene & Axes ---
        # Configure the 3D space
        axes = ThreeDAxes(
            x_range=[-2, 4, 1],
            y_range=[-2, 4, 1],
            z_range=[0, 15, 5],
            x_length=7,
            y_length=7,
            z_length=5,
        ).add_coordinates()

        # Labels for the axes
        labels = axes.get_axis_labels(
            x_label="w", y_label="b", z_label="Loss"
        )

        # --- 2. Define the Loss Landscape ---
        # We simulate the MSE landscape of a linear regression
        # Target: w=2, b=1. Function roughly: (w-2)^2 + (b-1)^2
        def loss_function(w, b):
            return 1.2 * (w - 2) ** 2 + 0.8 * (b - 1) ** 2

        # Create the 3D Surface
        loss_surface = Surface(
            lambda u, v: axes.c2p(u, v, loss_function(u, v)),
            u_range=[-2, 4],
            v_range=[-2, 4],
            resolution=(30, 30),
            should_make_jagged=False
        )

        # Style the surface (Semi-transparent blue)
        loss_surface.set_style(fill_opacity=0.3)
        loss_surface.set_fill_by_checkerboard(BLUE, BLUE_E, opacity=0.3)
        loss_surface.set_shade_in_3d(True)

        # --- 3. Generate Gradient Descent Path Data ---
        # Instead of importing PyTorch (which might slow down Manim rendering),
        # we simulate the math of the descent manually here.
        # w_grad = 2.4 * (w - 2)
        # b_grad = 1.6 * (b - 1)

        w_curr, b_curr = -1.5, -1.5  # Start far away on the rim
        lr = 0.1
        path_points = []

        # Calculate steps
        for _ in range(25):
            loss = loss_function(w_curr, b_curr)
            # Convert abstract coordinates to Manim 3D point
            point_3d = axes.c2p(w_curr, b_curr, loss)
            path_points.append(point_3d)

            # Update (Gradient Descent Step)
            w_curr -= lr * 2.4 * (w_curr - 2)
            b_curr -= lr * 1.6 * (b_curr - 1)

        # --- 4. Animation Sequence ---

        # A. Setup Camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        # B. Draw Axes and Surface
        self.add(axes, labels)
        self.play(Create(loss_surface), run_time=2)
        self.wait(0.5)

        # C. Initialize the "Neuron" (Dot) at start position
        start_point = path_points[0]
        dot = Dot3D(point=start_point, radius=0.1, color=RED)
        self.add(dot)

        # D. Create the trace (History path)
        trace = TracedPath(dot.get_center, stroke_color=YELLOW, stroke_width=4)
        self.add(trace)

        # E. Animate the Descent
        # We move the dot through every point in our calculated history
        for target_point in path_points[1:]:
            self.play(
                dot.animate.move_to(target_point),
                run_time=0.2,
                rate_func=linear
            )

        # F. Final Camera Rotation to admire the result
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)