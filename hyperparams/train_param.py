class TrainParam:
    """
    Hyper-parameters for main.py
    """
    n_step = 10
    draw_step = 1

    # paths
    output_dir = 'output'  # output root folder
    iter_img_dir = 'output/iter_images'  # folder to save iteration images for debugging
    stylized_img_dir = 'output/stylized_images'  # folder to save stylized video frames
    video_frames_dir = 'output/video_frames'  # folder to save video frames
    consistent_frames_dir = 'output/consistent_frames' # folder to save consistent video frames
    iter_consistent_frames_dir = 'output/iter_consistent_frames_dir' # folder to save iteration consistent video images for debugging 

    # optic flow

    # whether to calculate optic flow for better effect.
    # This may take a lot of time on CPU
    use_optic_flow = False

    video_optic_flow_dir = 'output/optic_flow'  # folder to save optic flow

    convergence_threshold = 0.01 # %change of loss for stopping training thereshold
    consistency_step = 10 # consistency 
    change_passdir_step = 5 # every x step to change pass direction
    check_frame_step = 33 # random frame idx to change
