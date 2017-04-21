import numpy as np
import scipy.misc
import cPickle

result_dir = "./result/samples/lsun_64_wgan_RMS/"
test_offset = 0

with open(result_dir+'test_fixed_prob_{}.pkl'.format(test_offset),'r') as infile:
    test_image_idxs, test_images, test_z_batches, save_result_g_loss, save_result_d_loss = cPickle.load(infile)

prob_image_pairs = [[] for i in range(64)]

for test_round_idx in range(64):
    path = result_dir+'test_fixed_round_{}{:02d}.png'.format(test_offset, test_round_idx)
    image_result = scipy.misc.imread(path)
    for row in range(8):
        for col in range(8):
            #(64*8*row, 64*col)
            image_idx = 8*row+col
            image = image_result[64*row:64*row+64, 64*col:64*col+64, :]
            prob = abs(save_result_d_loss[test_round_idx])
            prob_image_pairs[image_idx].append((prob, image))

with open(result_dir+'test_fixed_prob_sorted_{}.txt'.format(test_offset),'w') as outfile:
    for image_idx in range(64):
        outfile.write('Image %d:\n'%image_idx)
        sorted_images = np.zeros((64*8, 64*8, 3),dtype=np.uint8)
        prob_image_pair_sorted = sorted(prob_image_pairs[image_idx])
        for row in range(8):
            for col in range(8):
                sorted_images[64*row:64*row+64, 64*col:64*col+64, :] = prob_image_pair_sorted[8*row+col][1]
                outfile.write('%.7f\t'%prob_image_pair_sorted[8*row+col][0])
            outfile.write('\n')
        outfile.write('\n')
        scipy.misc.imsave(result_dir+'test_fixed_{}{:02d}_sorted.png'.format(test_offset, image_idx), sorted_images)



