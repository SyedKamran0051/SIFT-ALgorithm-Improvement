from skimage.io import imread
from sift import SIFT

import argparse
import pickle
import os
from os.path import isdir
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PySIFT')
	parser.add_argument('--input', type=str, dest='input_fname')
	parser.add_argument('--output', type=str, dest='output_prefix', help='The prefix for the kp_pyr and feat_pyr files generated')
	parser.add_argument('--mono', type=bool, dest='mono', default=False)
	args = parser.parse_args()

	# If you want to pass a mono view image rather than a stereo image pass --mono True arg
	if (args.mono):
		im = imread(args.input_fname)
		l = int(im.shape[1]/2)
		im = im[:, :l]
		print("Read image mono successfully!")
	
	# For a stereo image 	
	else: 
		im = imread(args.input_fname)
		print("Read image stereo successfully!")


	
	sift_detector = SIFT(im)
	_ = sift_detector.get_features()
	kp_pyr = sift_detector.kp_pyr

	if not isdir('results'):
		os.mkdir('results')

	# Dumping the pickle file created
	pickle.dump(sift_detector.kp_pyr, open('results/%s_kp_pyr.pkl' % args.output_prefix, 'wb'))
	pickle.dump(sift_detector.feats, open('results/%s_feat_pyr.pkl' % args.output_prefix, 'wb'))

	# For displaying the resultant image 
	_, ax = plt.subplots(1, sift_detector.num_octave)
	
	# For visualizing the scatter points over the output image 
	for i in tqdm(range(sift_detector.num_octave)):
		ax[i].imshow(im)

		scaled_kps = kp_pyr[i] * (2**i)
		ax[i].scatter(scaled_kps[:,0], scaled_kps[:,1], c='r', s=2.5)

	plt.show()
