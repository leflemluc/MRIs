import nibabel as nib
import numpy as np
import pylab as py
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import sys
dpi = 1000



def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

def display(PATH):
    epi_img = nib.load(PATH)
    epi_img_data = epi_img.get_fdata()
    cut_x = epi_img_data.shape[0] // 2
    cut_y = epi_img_data.shape[1] // 2
    cut_z = epi_img_data.shape[2] // 2
    slice_0 = epi_img_data[cut_x, :, :]
    slice_1 = epi_img_data[:, cut_y, :]
    slice_2 = epi_img_data[:, :, cut_z]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for EPI image")


def display_movie_x(PATH, save_path, title):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title, artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=5, metadata=metadata)

    epi_img = nib.load(PATH)
    epi_img_data = epi_img.get_fdata()

    fig, axes = plt.subplots(1, 1)

    with writer.saving(fig, save_path, 100):
        for i in range(0, 256, 3):
            print(i)
            slice = epi_img_data[i, :, :]
            axes.imshow(slice.T, cmap="gray", origin="lower")
            writer.grab_frame()



def display_movie_y(PATH, save_path, title):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title, artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=5, metadata=metadata)

    epi_img = nib.load(PATH)
    epi_img_data = epi_img.get_fdata()

    fig, axes = plt.subplots(1, 1)

    with writer.saving(fig, save_path, 100):
        for i in range(0, 256, 3):
            print(i)
            slice = epi_img_data[:, i, :]
            axes.imshow(slice.T, cmap="gray", origin="lower")
            writer.grab_frame()

def display_movie_z(PATH, save_path, title):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title, artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=5, metadata=metadata)

    epi_img = nib.load(PATH)
    epi_img_data = epi_img.get_fdata()

    fig, axes = plt.subplots(1, 1)

    with writer.saving(fig, save_path, 100):
        for i in range(0, 256, 3):
            print(i)
            slice = epi_img_data[:, :, i]
            axes.imshow(slice.T, cmap="gray", origin="lower")
            writer.grab_frame()



if __name__=='__main__':



