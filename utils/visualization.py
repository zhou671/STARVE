import moviepy.editor as mpy
import cv2


def resize_video(clip, width=None, height=None, target_clip=None, fps=None):
    """
    Resize a video clip.
    The resize function of moviepy is not as good.
    Either specify (`width`, `height`) or use the size of `target_clip`.
    :param clip: video clip to resize
    :param width: target width
    :param height: target height
    :param target_clip: use the size of `target_clip`
    :param fps: frame per second
    :return:
        clip: Resized video clip.
    """
    assert (width is not None and height is not None) or target_clip is not None, "Specify size."
    if target_clip:
        width, height = target_clip.size
    if fps is None:
        fps = clip.fps

    frames = []
    for frame in clip.iter_frames(fps=fps, dtype="uint8"):
        frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        frame = mpy.ImageClip(frame).set_duration(1 / fps)
        frames.append(frame)
    clip = mpy.concatenate_videoclips(frames, method="compose")
    clip = clip.set_fps(fps)

    return clip


def overlay_img_on_video(img_clip, video_clip):
    """
    Overlay an image on a video clip.
    :param img_clip: image clip
    :param video_clip: video clip
    :return:
        img_clip: Modified image clip.
        combine_clip: Combined clip with image and video.
    """
    img_clip = (img_clip
                .set_duration(video_clip.duration)
                .resize(height=video_clip.size[1] // 4)
                .set_pos(("right", "bottom")))
    combine_clip = mpy.CompositeVideoClip([video_clip, img_clip])

    return img_clip, combine_clip


def compare_videos_transition(src_video_path, style_img_path, stylized_video_path, save_path=""):
    """
    Use a smooth transition to show the difference between source video and stylized video.
    :param src_video_path: path to the original video
    :param style_img_path: path to the style image
    :param stylized_video_path: path to the stylized video
    :param save_path: path to save the output video with transition
    :return:
        final: The output video with transition.
    """
    src_video = mpy.VideoFileClip(src_video_path)
    style_img = mpy.ImageClip(style_img_path)
    stylized_video = mpy.VideoFileClip(stylized_video_path)
    src_aidio = src_video.audio

    stylized_video = resize_video(stylized_video, target_clip=src_video)

    # The first `transition_duration` seconds is for transition.
    transition_duration = 2.5
    frames = []
    fps = stylized_video.fps
    rate = 1 / (transition_duration * stylized_video.fps)
    for i, (src_f, target_f) in \
            enumerate(
                zip(src_video.subclip(0, transition_duration).iter_frames(fps=fps, dtype="uint8"),
                    stylized_video.subclip(0, transition_duration).iter_frames(fps=fps, dtype="uint8"))):
        compose = cv2.addWeighted(src_f, 1 - rate * i, target_f, rate * i, 0)
        frame = mpy.ImageClip(compose).set_duration(1 / stylized_video.fps)
        frames.append(frame)
    transition_video = mpy.concatenate_videoclips(frames, method="compose")

    # Later, resize the stylized video.
    stylized_video = stylized_video.subclip(transition_duration)

    final = mpy.concatenate_videoclips([transition_video, stylized_video])
    final = final.set_fps(fps)
    _, final = overlay_img_on_video(style_img, final)
    final.audio = src_aidio.subclip(0, final.duration)

    if save_path:
        final.write_videofile(save_path)

    return final


def compare_videos_stack(src_video_path, style_img_path, stylized_video_path, save_path=""):
    """
    Stack two videos on the same row
    to show the difference between source video and stylized video.
    :param src_video_path: path to the original video
    :param style_img_path: path to the style image
    :param stylized_video_path: path to the stylized video
    :param save_path: path to save the output video with transition
    :return:
        final: The output video with transition.
    """
    src_video = mpy.VideoFileClip(src_video_path)
    style_img = mpy.ImageClip(style_img_path)
    stylized_video = mpy.VideoFileClip(stylized_video_path)

    style_img, src_video = overlay_img_on_video(style_img, src_video)

    # align video length
    if stylized_video.duration > src_video.duration:
        stylized_video = stylized_video.subclip(0, src_video.duration)
    stylized_video = resize_video(stylized_video, src_video.size[0], src_video.size[1])

    final = mpy.clips_array([[src_video, stylized_video]])
    if save_path:
        final.write_videofile(save_path)

    return final


if __name__ == '__main__':
    the_source_video = r"../demo/short_video.mp4"
    the_style_image = r"../demo/mrbean.png"
    the_output_video = r"../output/stylized_short_video tv10iters.mp4"
    the_save_path = r"../output/transition.mp4"

    compare_videos_stack(the_source_video, the_style_image, the_output_video, the_save_path)
