import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc
import skimage.transform
import imageio.v2 as imageio

import argparse
from sklearn.neighbors import NearestNeighbors
from skimage.transform import pyramid_gaussian

# Define a function to resize an image with anti-aliasing
imresize = lambda x, shape: skimage.transform.resize(x, shape, anti_aliasing=True, mode='constant')

# Define a function to convert an RGB image to grayscale
def rgb2gray(rgb):
    if rgb.ndim == 2 or (rgb.ndim == 3 and rgb.shape[2] == 1):  # Grayscale image
        return rgb if rgb.ndim == 2 else rgb[:, :, 0]
    elif rgb.ndim == 3 and rgb.shape[2] == 2:  # Grayscale with alpha channel
        return rgb[:, :, 0]  # Assuming the first channel is the grayscale channel
    elif rgb.ndim == 3 and rgb.shape[2] == 3:  # RGB image
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        raise ValueError("Unsupported image format!")

# Define a function to read an image from a file
def read_image(filename):
    I = imageio.imread(filename)
    I = np.array(I, dtype=np.float32) / 255.0
    return I

# Define a function to write an image to a file
def write_image(I, filename):
    # Normalize and convert to uint8
    IRet = I * 255.0
    IRet[IRet > 255] = 255
    IRet[IRet < 0] = 0
    IRet = np.array(IRet, dtype=np.uint8)

    # Check if the image is single-channel and convert it to 3-channel if needed
    if IRet.ndim == 2 or (IRet.ndim == 3 and IRet.shape[2] == 1):
        IRet = np.stack((IRet, IRet, IRet), axis=-1) if IRet.ndim == 2 else np.repeat(IRet, 3, axis=2)

    # Save the image
    imageio.imwrite(filename, IRet)

# Define a function to get patches from an image
def getPatches(I, dim):
    shape = np.array(I.shape * 2)
    strides = np.array(I.strides * 2)
    W = np.asarray(dim)
    shape[I.ndim:] = W
    shape[:I.ndim] -= W - 1
    if np.any(shape < 1):
        raise ValueError('Window size %i is too large for image' % dim)
    P = np.lib.stride_tricks.as_strided(I, shape=shape, strides=strides)
    P = np.reshape(P, [P.shape[0], P.shape[1], dim * dim])
    return P

# Define a function to get causal patches from an image
def getCausalPatches(I, dim):
    P = getPatches(I, dim)
    k = int((dim * dim - 1) / 2)
    P = P[:, :, 0:k]
    return P

# Define a function to find the best coherent match for a pixel
def getCoherenceMatch(X, x0, BpLidx, dim, i, j):

    k = int((dim * dim - 1) / 2)
    M = X.shape[0]
    N = X.shape[1]
    minDistSqr = np.inf
    idxmin = [-1, -1]
    [dJ, dI] = np.meshgrid(np.arange(dim), np.arange(dim))
    dI = np.array(dI.flatten()[0:k], dtype=np.int64) - (dim - 1) / 2
    dJ = np.array(dJ.flatten()[0:k], dtype=np.int64) - (dim - 1) / 2
    # TODO: Vectorize code below
    for n in range(dI.size):
        # Indices of pixel picked for neighbor
        ni = BpLidx[int(dI[n] + i), int(dJ[n] + j)][0]
        nj = BpLidx[int(dI[n] + i), int(dJ[n] + j)][1]
        if ni == -1 or nj == -1:
            continue
        ni = int(ni - dI[n])
        nj = int(nj - dJ[n])
        if ni < 0 or nj < 0 or ni >= M or nj >= N:
            continue
        x = X[ni, nj, :]
        distSqr = np.sum((x - x0) ** 2)
        if distSqr < minDistSqr:
            minDistSqr = distSqr
            idxmin = [ni, nj]
    return (idxmin, minDistSqr)

# Define the main image analogies function
def do_image_analogies(input_A_path, input_Ap_path, input_B_path,output_Bp_path, Kappa=0.0, NLevels=3, KCoarse=5, KFine=5, n_jobs=None, debugImages=False):
    A = read_image(input_A_path)
    Ap = read_image(input_Ap_path)
    B = read_image(input_B_path)
    # Make image pyramids
    AL = tuple(pyramid_gaussian(A, NLevels, downscale=2))
    ApL = tuple(pyramid_gaussian(Ap, NLevels, downscale=2))
    BL = tuple(pyramid_gaussian(B, NLevels, downscale=2))
    BpL = []  # B' values in pyramid
    BpLidx = []  # Indices of nearest neighbors at each pixel in pyramid
    print("BL:")
    for i in range(len(BL)):
        print(BL[i].shape)
        BpL.append(np.zeros(BL[i].shape))
        BpLidx.append(-1 * np.ones((BL[i].shape[0], BL[i].shape[1], 2), dtype=int))
    print("AL:")
    for i in range(len(AL)):
        print(AL[i].shape)

    # Do multiresolution synthesis
    for level in range(NLevels, -1, -1):
        KSpatial = KFine
        if level == NLevels:
            KSpatial = KCoarse
        # Step 1: Make features
        APatches = getPatches(rgb2gray(AL[level]), KSpatial)
        ApPatches = getCausalPatches(rgb2gray(ApL[level]), KSpatial)
        X = np.concatenate((APatches, ApPatches), 2)
        B2 = None
        Bp2 = None
        if level < NLevels:
            # Use multiresolution features
            A2 = imresize(AL[level + 1], AL[level].shape)
            Ap2 = imresize(ApL[level + 1], ApL[level].shape)
            A2Patches = getPatches(rgb2gray(A2), KSpatial)
            Ap2Patches = getPatches(rgb2gray(Ap2), KSpatial)
            X = np.concatenate((X, A2Patches, Ap2Patches), 2)
            B2 = imresize(BL[level + 1], BL[level].shape)
            Bp2 = imresize(BpL[level + 1], BpL[level].shape)
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=n_jobs).fit(
            np.reshape(X, [X.shape[0] * X.shape[1], X.shape[2]]))

        # Step 2: Fill in the first few scanLines to prevent the image
        # from getting corrupted in the beginning
        if level == NLevels:
            I = np.array(ApL[level] * 255, dtype=np.uint8)
            I = imresize(I, BpL[level].shape)
            BpL[level] = np.array(I / 255.0, dtype=np.float64)
        else:
            I = np.array(BpL[level + 1] * 255, dtype=np.uint8)
            I = imresize(I, BpL[level].shape)
            BpL[level] = np.array(I / 255.0, dtype=np.float64)

        # Step 3: Fill in the pixels in scanline order
        d = int((KSpatial - 1) / 2)
        for i in range(d, BpL[level].shape[0] - d):
            print(i)
            for j in range(d, BpL[level].shape[1] - d):
                # Make the feature at this pixel
                # Full patch B
                BPatch = rgb2gray(BL[level][i - d:i + d + 1, j - d:j + d + 1, :])
                # Causal patch B'
                BpPatch = rgb2gray(BpL[level][i - d:i + d + 1, j - d:j + d + 1, :]).flatten()
                BpPatch = BpPatch[0:int((KSpatial * KSpatial - 1) / 2)]
                F = np.concatenate((BPatch.flatten(), BpPatch.flatten()))

                if level < NLevels:
                    # Use multiresolution features
                    BPatch = rgb2gray(B2[i - d:i + d + 1, j - d:j + d + 1, :])
                    BpPatch = rgb2gray(Bp2[i - d:i + d + 1, j - d:j + d + 1, :])
                    F = np.concatenate((F, BPatch.flatten(), BpPatch.flatten()))
                # Find index of most closely matching feature point in A
                dist, idx = nn.kneighbors(F[None, :])
                idx = int(idx[0][0]) if idx.shape[1] == 1 else int(idx[0])
                distSqr = dist ** 2
                idx = np.unravel_index(idx, (X.shape[0], X.shape[1]))
                if Kappa > 0:
                    # Compare with coherent pixel
                    (idxc, distSqrc) = getCoherenceMatch(X, F, BpLidx[level], KSpatial, i, j)
                    fac = 1 + Kappa * (2.0 ** (level - NLevels))
                    if distSqrc < distSqr * fac * fac:
                        idx = idxc
                BpLidx[level][i, j, :] = idx
                BpL[level][i, j, :] = ApL[level][idx[0] + d, idx[1] + d, :]
            if i % 20 == 0 and debugImages:
                write_image(BpL[level], "%i.png" % level)
        if debugImages:
            plt.subplot(122)
            plt.imshow(BpLidx[level][:, :, 0], cmap='Spectral')
            plt.title("Y")
            plt.subplot(121)
            plt.imshow(BpLidx[level][:, :, 1], cmap='Spectral')
            plt.title("X")
            plt.savefig("%i_idx.png" % level, bbox_inches='tight')
    write_image(BpL[0], output_Bp_path)