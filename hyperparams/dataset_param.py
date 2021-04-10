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
