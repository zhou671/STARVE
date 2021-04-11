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
    video_optic_flow_dir = 'output/optic_flow'  # folder to save optic flow
