from god_config import *
import glob
import matplotlib.pyplot as plt
from scipy import misc
import pickle
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

def create_loss_figure():
    # check log folder exist or not
    if not os.path.exists(LOGS_PATH):
        print ("Cannot find log folder.")
        return
    log_sub_paths = glob.glob(LOGS_PATH+'/*')

    # check if there exist log files
    if len(log_sub_paths) == 0:
        print ("There are no log files.")
        return

    for sub_path in log_sub_paths:
        name = sub_path.split('/')[-1]
        loss_files = glob.glob(sub_path+'/*pkl')
        losses = []
        loss_param = []
        for lf in loss_files:
            with open(lf, 'rb') as f:
                cur_loss = pickle.load(f)
                losses.append(cur_loss)
                tmp = (lf.split('/')[-1]).split('_')[0]
                loss_param.append(tmp)
        x = [i for i in range(len(losses[0]))]
        for i in range(len(losses)):
            plt.plot(x, losses[i], label='lr='+loss_param[i], linewidth=3)
        plt.yscale('log')
        plt.xlabel('epochs (*100)', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend()
        plt.savefig(os.path.join(FIG_DIR,'adam_'+name+'_lr.pdf'), bbox_inches='tight')
        plt.gcf().clear()
create_loss_figure()

