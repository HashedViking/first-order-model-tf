import math
import tensorflow as tf
from tqdm import tqdm

from scipy.spatial import ConvexHull
import torch
import numpy as np

#General inference scheme:
#Step 0: determine batch size based on available memory
#Step 1: get kp_source
#Step 2: get kp_driving in batches
#Step 3: process kp_driving
#Step 4: get predictions in batches
def animate(source_image, driving_video, generator, kp_detector, process_kp_driving, batch_size=4, relative=True, adapt_movement_scale=True):
    import time
    start_time = time.time()

    l = len(driving_video)
    source_image = tf.convert_to_tensor(source_image,'float32')
    
    tf.profiler.experimental.start('./log')
    
    #Step 1: get kp_source
    kp_source = kp_detector(source_image)

    kp_detector_time = time.time() - start_time
    print("--- kp_source ---\n--- %s seconds ---" % kp_detector_time)

    #Step 2: get kp_driving in batches
    kp_driving = []
    for i in tqdm(range(math.floor(l/batch_size))):
        start = i*batch_size
        end = (i+1)*batch_size
        driving_video_tensor = tf.convert_to_tensor(driving_video[start:end])
        kp_driving.append(kp_detector(driving_video_tensor))
    kp_driving = tf.concat(kp_driving, 0)    
    del driving_video

    kp_driving_time = time.time() - start_time - kp_detector_time
    print("--- kp_driving ---\n--- %s seconds ---" % kp_driving_time)

    #Step 3: process kp_driving
    kp_driving = process_kp_driving(kp_driving,kp_source,relative,adapt_movement_scale)

    process_kp_driving_time = time.time() - start_time - kp_detector_time - kp_driving_time
    print("--- process_kp_driving ---\n--- %s seconds ---" % process_kp_driving_time)

    #Step 4: get predictions in batches
    predictions = []
    for i in tqdm(range(math.floor(l/batch_size))):
        start = i*batch_size
        end = (i+1)*batch_size
        kp_driving_tensor = kp_driving[start:end]
        # print(f'\nBatches {(i)*batch_size}/{l}')
        generator_start_time = time.time()
        predictions.append(generator([source_image,kp_driving_tensor,kp_source]))
        generator_time = time.time() - generator_start_time
        print("--- generator ---\n--- %s seconds ---" % generator_time)
    tf.profiler.experimental.stop()
    print(len(predictions))
    overall_time = time.time() - start_time
    print("--- overall_time ---\n--- %s seconds ---" % overall_time)
    return tf.concat(predictions,0).numpy()

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new
