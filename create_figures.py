from god_config import *
import glob
import matplotlib.pyplot as plt
from scipy import misc

NUM_OF_IMAGE_COL = 8


def create_recons_image_evolution(show_image=False):
    sub_paths = glob.glob(RECONS_IMAGE_PATH+'/*')
    sub_paths.sort()
    f, axarr = plt.subplots(len(sub_paths), NUM_OF_IMAGE_COL, gridspec_kw = {'wspace':0, 'hspace':0}, figsize= (9, 9))
    for i in range(len(sub_paths)):
        layer_name = sub_paths[i].split('/')[-1]
        images = glob.glob(sub_paths[i]+'/*')
        img_number = [int((img.split('/')[-1]).split('.')[0]) for img in images]
        sorted_idx = sorted(range(len(img_number)), key=lambda k:img_number[k])
        img_number = sorted(img_number)
        images = [images[i] for i in sorted_idx]
        if len(images) > 0:
            interval = int(len(images)/NUM_OF_IMAGE_COL)
            for j in range(NUM_OF_IMAGE_COL):
                im = misc.imread(images[j*interval])
                if len(sub_paths) > 1:
                    ax = axarr[i, j]
                elif len(sub_paths) == 1:
                    ax = axarr[j]
                ax.imshow(im)
                ax.tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False
                )
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlabel(str(img_number[j*interval]), fontsize=12)
                if j == 0:
                    ax.set_ylabel(layer_name, rotation=90, fontsize=12)
    if show_image:
        plt.show()
    if not os.path.exists(FIG_DIR):
        os.mkdir(FIG_DIR)
    plt.savefig(os.path.join(FIG_DIR, 'reconstructed_images.pdf'))

create_recons_image_evolution()

