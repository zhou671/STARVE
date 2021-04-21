from os.path import dirname, join, isdir, isfile, basename, splitext
from os import listdir
from subprocess import Popen, PIPE

frame_folder = 'video_frames'
flow_folder = 'optic_flow'
forward_flow_name = join(flow_folder, "forward_{}_{}.flo")
backward_flow_name = join(flow_folder, "backward_{}_{}.flo")
# reliable_file_name = join(flow_folder, "reliable_{}_{}.pgm")

# forward flow
files = sorted([int(f[:-4]) for f in listdir(frame_folder) if isfile(join(frame_folder, f))])
for k in [1, 2, 5]: # LossParam.J
  for i in range(len(files)-k):
    img1 = join(frame_folder, str(files[i])+".jpg")
    img2 = join(frame_folder, str(files[i+k])+".jpg")
    # file_name = forward_flow_name.format(str(files[i]), str(files[i+k]))
    file_name = backward_flow_name.format(str(files[i+k]), str(files[i]))
    # populate list of arguments
    args = ["generate_flow.py"]
    # for opt, optname in zip("--img1 --img2 --flow".split(), (img1+" "+img2+" "+file_name).split()):
    for opt, optname in zip("--img1 --img2 --flow".split(), (img2+" "+img1+" "+file_name).split()):
        args.extend([opt, optname])

    # run script
    p = Popen(['python'] + args, stdout=PIPE)
    p.stdout.close()
    p.wait()

