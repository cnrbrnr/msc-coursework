from manim import *

class CircleAnimation(Scene):
    def construct(self):
        # Create a circle
        circle = Circle()

        # Animate the circle onto the frame
        self.play(Create(circle))
        self.wait(1)  # Wait for a moment before ending the animation
