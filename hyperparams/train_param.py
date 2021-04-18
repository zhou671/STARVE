class TrainParam:
    """
    Hyper-parameters for main.py
    """
    n_step = 50  # number of iterations for one frame
    draw_step = 100  # draw intermediate result for debugging

    # paths
    output_dir = 'output'  # output root folder
    iter_img_dir = 'output/iter_images'  # folder to save iteration images for debugging
    stylized_img_dir = 'output/stylized_images'  # folder to save stylized video frames
    video_frames_dir = 'output/video_frames'  # folder to save video frames

    # optic flow

    # whether to calculate optic flow for better effect.
    # This means to use the temporal loss.
    # This may take a lot of time on CPU
    use_optic_flow = True
    use_deep_matching_gpu = True  # whether to use the GPU version of DeepMatching
    video_optic_flow_dir = 'output/optic_flow'  # folder to save optic flow

    # multi-pass
    n_passes = 1  # number of passes, the first pass will not use blend factor
