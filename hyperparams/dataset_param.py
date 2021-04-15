class DatasetParam:
    """
    Hyper-parameters for utils/dataset.py
    """
    img_w = 450  # image width
    img_h = 350  # image height

    use_video = True  # Style transfer with a video or an image. True: video; False: image
    content_img_path = r'demo/dog.jpg'
    video_path = r'demo/short_video.mp4'
    style_img_path = r'demo/mrbean.png'

    img_fmt = 'jpg'  # frame image format
    video_fps = 24  # select `video_fps` frames per second

    # method to calculate optic flow
    # 'dm_df2': DeepMatching + DeepFlow2
    # 'df2': DeepFlow2 only
    optic_flow_method = 'dm_df2'

    # method to initialize the stylized image
    # 'image': use the current video frame image
    # 'image_flow_warp': warp the previous stylized image with optic flow
    # 'random': normal distribution
    init_generated_image_method = 'image'
