import tensorflow as tf
import librosa
import os
from IPython.display import Audio, display
import numpy as np
import matplotlib.pyplot as plt
import math

# This code is based on code by Dmitry Uylanov. Much of this code is simply a refactoring of Uylanovs code.
# This algorithm improves slightly upon Uylanov's method by implementing ideas from Eric Grinstein et al, along with a few improvements
# discovered by myself (Wesley Slawson) and a few collaborators (Domenic Cusanelli, Jonathan Satterfield, Matthew Crow).
# Contributions:
# Wesley Slawson: refactoring of Uylanov code, discovery of all improvements, and implementation of all improvements (excepting the kernel filter)
# Jonathan Satterfield: implementation of kernel filter (filter_limiter method)
# Domenic Cusanelli, Matthew Crow: General help with project research
#
# The improvements made on the Uylanov algorithm are described on the Github Page for this project, which can be found at
# https://wslawson1.github.io/AudioStyleTransfer/
#
# Dmitry Uylanov's Audio Style Transfer project webpage:
# https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/
#
# Grinstein Paper:
# https://arxiv.org/pdf/1710.11385.pdf
#

# general flow of algorithm:
# Read audio in
# generate spectogram
# compute features/ gram matrices
# optimize

# returns a version of the kernel which limits nonzero values to contiguous areas of kernel
# corresponding to 1, 2, or 3 octave ranges
def filter_limiter(kernel, sample_rate, n_channels, removal=0):
    limiter = np.full_like(kernel, removal)
    for height in range(kernel.shape[0]):
        for width in range(kernel.shape[1]):
            for filter in range(kernel.shape[3]):
                lower_bound = 2**(np.random.random()*8 + 1) * 11
                upper_bound = lower_bound * 2**np.random.randint(1,3)
                max_freq = sample_rate / 2
                resolution = max_freq / n_channels
                cells = round((upper_bound - lower_bound)/resolution)
                offset = round(lower_bound / resolution)
                limiter[height,width,offset:offset+cells,filter] = 1
    limitedKernel = np.multiply(kernel, limiter)
    return limitedKernel

# returns logorithmically normalized spectogram of time series
def getSpectogram( timeSeries, n_fft, hop_length ):
    S = librosa.core.stft( timeSeries, n_fft=n_fft, hop_length=hop_length )
    S = np.log1p(np.abs(S[:,:]))  
    return S

# generates kernel with random values in specified shape
def generateRandKernel( n_channels, n_filters, width ):
    std = np.sqrt(2) * np.sqrt(2.0 / ((n_channels + n_filters) * 11))
    kernel = np.random.randn(1, width, n_channels, n_filters)*std
    return kernel

# computes the feature map of inData using specified kernel
def computeFeatureMap(inData, kernel, n_channels, n_samples ):
    g = tf.Graph()
    featureMap = None
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        inputs = tf.placeholder('float32', [1, 1, n_samples, n_channels], name="inputs" )

        kernel_tensor = tf.constant( kernel, name="kernel", dtype='float32' )

        convolution = tf.nn.conv2d(
            inputs,
            kernel_tensor,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="convolution"
        )

        featureMapNet = tf.nn.relu( convolution )

        featureMap = featureMapNet.eval( feed_dict={inputs: inData } ) 
    
    return featureMap

# computes gram matrix for specified feature map
def gramMatrix( featureMap, n_filters, n_samples ):
    features = np.reshape(featureMap, (-1, n_filters ) )
    gram = np.matmul( features.T, features ) / n_samples
    return gram

# optimizes input spectogram based on two-fold loss function 
# minimizes loss between feature map and previously computed content feature map
# and between gram matrix and style gram matrix
def optimizeInput( inputSpectogram, kernel, content_Features, style_gram, n_samples, contentLossWeight, max_iterations ):
    result = None

    with tf.Graph().as_default():
        contentVariable = tf.Variable( inputSpectogram, name="contentVariable" )
        kernel_tensor = tf.constant( kernel, name="kernel", dtype='float32')

        convolution = tf.nn.conv2d(
            contentVariable,
            kernel_tensor,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="convolution"
        )

        featureMapNet = tf.nn.relu( convolution )

        content_loss = contentLossWeight * 2 * tf.nn.l2_loss( featureMapNet - content_Features )
        
        style_loss = 0
        _, height, width, number = map(lambda i: i.value, featureMapNet.get_shape())

        reshapedFeatures = tf.reshape( featureMapNet, ( -1, number ) )
        contentGram = tf.matmul( tf.transpose( reshapedFeatures ), reshapedFeatures ) / n_samples
        style_loss = 2 * tf.nn.l2_loss( contentGram - style_gram )

        totalLoss = content_loss + style_loss

        opt = tf.contrib.opt.ScipyOptimizerInterface(
            totalLoss, method='L-BFGS-B', options={'maxiter': max_iterations }
        )

        with tf.Session() as sess:
            sess.run( tf.global_variables_initializer() )

            print('Started optimization.')
            opt.minimize( sess )

            print( 'Final loss:' )
            print( totalLoss.eval() )
            result = contentVariable.eval()
    
    return result

# performs phase reconstruction based on griffin-lim algorithm
def reconstructPhase( spectogram, n_fft, hop_length ):
    p = 2 * np.pi * np.random.random_sample(spectogram.shape) - np.pi
    for i in range(500):
        S = spectogram * np.exp(1j*p)
        x = librosa.istft(S, hop_length)
        p = np.angle(librosa.stft(x, n_fft, hop_length))
    return x

# displays spectogram and saves spectogram image if provided a filename
def displaySpectogram( spectogram, name="" ):
    plt.figure(figsize=(15,5))
    plt.title('spectogram')
    plt.imshow( spectogram )
    plt.show()

    if ( name != "" ):
        plt.savefig( name )

# constants for a given run of program
n_fft=512
hop_length=int( n_fft/8 )
n_filters=512
contentLossWeight=1e-2
max_iterations=300
kernelWidth = 5
duration = 1

contentFileName="HandPercussion"
styleFileName="PianoChords"


#duration is in seconds
contentTimeSeries, sampleRate = librosa.load("inputs/" + contentFileName + ".wav", duration=duration )
styleTimeSeries, sampleRate = librosa.load("inputs/" + styleFileName + ".wav", duration=duration)

# names output for convenience in identifying the settings used to produce the output
idString = contentFileName + "X" + styleFileName + "nfft"+str(n_fft)+"_hl"+str(hop_length)+"_filt"+str(n_filters)+"alpha"+str(contentLossWeight)+"_itr"+str(max_iterations)+"NONOISE"
outFilenameWav="outputAudioNew/" + idString + ".wav"
outFilenamePng="outputImgNew/" + idString + ".png"

#retrieve spectograms
contentSpectogram = getSpectogram( contentTimeSeries, n_fft=n_fft, hop_length=hop_length)
styleSpectogram = getSpectogram( styleTimeSeries, n_fft=n_fft, hop_length=hop_length)

#ensure content and style spectogram have the same shape
n_channels = contentSpectogram.shape[0]
n_samples = contentSpectogram.shape[1]
styleSpectogram = styleSpectogram[ :n_channels, :n_samples ]

displaySpectogram( styleSpectogram )
print(contentSpectogram.shape)

# make arrays contiguous 
contentSpectogramCont = np.ascontiguousarray( contentSpectogram.T[None, None, :, :] )
styleSpectogramCont = np.ascontiguousarray( styleSpectogram.T[None, None, :, :] )

# generate random kernel and apply filter
kernel = generateRandKernel( n_channels=n_channels, n_filters=n_filters, width=kernelWidth )
kernel = filter_limiter(kernel, sampleRate, n_channels=n_channels)

# compute feature maps and style gram matrix, then optimize
styleFeatures = computeFeatureMap( styleSpectogramCont, kernel=kernel, n_channels=n_channels, n_samples=n_samples )
contentFeatures = computeFeatureMap( contentSpectogramCont, kernel=kernel, n_channels=n_channels, n_samples=n_samples )
styleGram = gramMatrix( styleFeatures, n_filters=n_filters, n_samples=n_samples )

result = optimizeInput( 
    contentSpectogramCont,
    kernel=kernel,
    content_Features=contentFeatures,
    style_gram=styleGram,
    n_samples=n_samples,
    contentLossWeight=contentLossWeight,
    max_iterations=max_iterations
)

# Perform phase reconstruction, write output
resultSpectogram = np.zeros_like( contentSpectogram )
resultSpectogram[:n_channels,:] = np.exp(result[0,0].T) - 1
resultTimeSeries = reconstructPhase( resultSpectogram, n_fft, hop_length)
librosa.output.write_wav( outFilenameWav, resultTimeSeries, sampleRate )

#displaySpectogram( resultSpectogram )

# display content, style, and result spectograms for comparison
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Content')
plt.imshow(contentSpectogram[:,:])
plt.subplot(1,3,2)
plt.title('Style')
plt.imshow(styleSpectogram[:,:])
plt.subplot(1,3,3)
plt.title('Result')
plt.imshow(resultSpectogram[:,:])
plt.savefig(outFilenamePng)
plt.show()