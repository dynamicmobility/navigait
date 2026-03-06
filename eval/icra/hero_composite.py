from dynamo_figures import CompositeImage, CompositeMode
import cv2

merger = CompositeImage(
    mode=CompositeMode.MAX_VARIATION,
    video_path=f'eval/icra/videos/NaviGait_rollout.mp4',
    start_t=0.0,
    end_t=20.0,
    skip_frame=100,
    alpha=0.6,
    disable_pbar=False  # Set to True to disable the progress bar
)

# Generate the composite
result = merger.merge_images()

# Save the result
cv2.imwrite(f'paper_plots/hero_walking.jpg', result)