import cv2
import os
import numpy as np
import random
import argparse
import pandas as pd
# from tqdm import tqdm

SEED = 10
random.seed(SEED)
np.random.seed(SEED)

# Dataset Parameters
img_size = 224
size = 15
question_size = 10 # 6 for one-hot vector of color, 1 for question type, 3 for question subtype
q_type_idx = 6
sub_q_type_idx = 7
nb_questions = 10
# Possibles Answers : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]

colors = [
    (0,0,255), # red
    (0,255,0), # green
    (255,0,0), # blue
    (0,156,255), # orange
    (128,128,128), # gray
    (0,255,255) # yellow
]

def center_generate(objects):
    '''Generate centers of objects'''
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name, c, shape, _ in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center

def build_sample():
    '''Returns an image (with its bbox and attributes) and the corresponding questions, programs and answers'''
    
    # Create objects
    objects = [] # [(color, center, shape, xmin, ymin, xmax, ymax), (...), ...]
    img = np.ones((img_size,img_size,3)) * 255
    for color_id, color in enumerate(colors):  
        center = center_generate(objects)
        if random.random()<0.5:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)

            # Get the rectangle corners (top-left, top-right, bottom-right, bottom-left)
            rect_corners = np.array([
                [start[0], start[1]],  # Top-left corner
                [end[0], start[1]],    # Top-right corner
                [end[0], end[1]],      # Bottom-right corner
                [start[0], end[1]]     # Bottom-left corner
            ], dtype=np.float32)

            # Generate a random angle between 0 and 360 degrees
            angle = random.uniform(0, 360)

            # Get the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), angle, 1.0)

            # Reshape rect_corners to (n, 1, 2) format for cv2.transform
            rect_corners = rect_corners.reshape((-1, 1, 2))

            # Apply rotation to the rectangle corners
            rotated_rect_corners = cv2.transform(rect_corners, rotation_matrix)

            # Extract the rotated points (now 2D array)
            rotated_rect_corners = rotated_rect_corners.reshape(-1, 2)

            # Draw the rotated rectangle (using fillPoly to simulate rotation)
            cv2.fillPoly(img, [rotated_rect_corners.astype(np.int32)], color)
            # cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id, center, 'r', (start[0], start[1], end[0], end[1]))) 
        else:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            objects.append((color_id, center, 'c', (start[0], start[1], end[0], end[1])))
    img = img / 255.
    sample = (img, objects)
    
    return sample

# def convert_sample(sample):
#     '''Converts question/answer vector to natural language questions and programs'''
    
#     img, objects, (rel_questions, rel_answers), (norel_questions, norel_answers) = sample
#     colors = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']
#     answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']
#     questions = rel_questions + norel_questions
#     answers = rel_answers + norel_answers
    
#     queries = []
#     programs = []
#     text_answers = []

#     for i, (question, answer) in enumerate(zip(questions, answers)):
#         query = f'Q{i}. '
#         color = colors[question.tolist()[0:6].index(1)]
        
#         # Non-relational questions
#         if question[q_type_idx] == 0:
#             if question[sub_q_type_idx] == 1:
#                 queries.append(f'What is the shape of the {color} object?')
#                 programs.append(f'filter {color} <nxt> query shape')
                
#             elif question[sub_q_type_idx+1] == 1:
#                 queries.append(f'Is there a {color} object on the left?')
#                 programs.append(f'filter {color} <nxt> query position <nxt> isLeft')
                
#             elif question[sub_q_type_idx+2] == 1:
#                 queries.append(f'Is there a {color} object on the top?')
#                 programs.append(f'filter {color} <nxt> query position <nxt> isTop')
            
#         # Relational questions
#         elif question[q_type_idx] == 1:
#             if question[sub_q_type_idx] == 1:
#                 queries.append(f'What is the closest shape to the {color} object?')
#                 programs.append(f'filter {color} <nxt> relate closest <nxt> query shape')
                
#             elif question[sub_q_type_idx+1] == 1:
#                 queries.append(f'What is the furthest shape from the {color} object?')
#                 programs.append(f'filter {color} <nxt> relate furthest <nxt> query shape')
                
#             elif question[sub_q_type_idx+2] == 1:
#                 queries.append(f'How many objects of the same shape as the {color} object are there?')
#                 programs.append(f'filter {color} <nxt> query shape <nxt> filter <nxt> count')
        
#         ans = answer_sheet[answer]
#         text_answers.append(ans)
        
#     return img, objects, queries, programs, text_answers
    
# def build_dataset(num_samples, data_dir, prefix='train'):
#     '''Builds a Full Dataset with Images, Detector Data, Attribute Data, Queries, Programs and Answers'''
    
#     # Generate Samples
#     samples = [build_sample() for _ in range(num_samples)]
    
#     # Init dataframes
#     img_det_df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
#     que2prog_df = pd.DataFrame(columns=['filename', 'answer', 'query_text', 'program_text'])
    
#     img_dir = os.path.join(data_dir, 'images')
#     shape_map = {'r': 'rectangle', 'c': 'circle'}
    
#     try:
#         os.makedirs(data_dir)
#     except:
#         pass
    
#     try:
#         os.makedirs(img_dir)
#     except:
#         pass
    
#     for i, sample in enumerate(tqdm(samples)):
#         # Get Data
#         img, objects, queries, programs, answers = convert_sample(sample)
        
#         # Save Image
#         filename = f'{i}.jpg'
#         img_path = os.path.join(img_dir, filename)
#         cv2.imwrite(img_path, img * 255)
        
#         # Append image data to dataframes
#         for obj in objects:
#             # Get object params
#             color_id, shape, bbox = obj[0], shape_map[obj[2]], obj[3]
#             new_row = pd.DataFrame([{'filename': filename, 
#                                             'width': img_size, 
#                                             'height': img_size, 
#                                             'class': 'obj', 
#                                             'xmin': bbox[0], 'ymin': bbox[1],
#                                             'xmax': bbox[2], 'ymax': bbox[3]}])
#             img_det_df = pd.concat([img_det_df, new_row], ignore_index=True)
        
#         # Append text data to dataframe
#         for answer, query, program in zip(answers, queries, programs):
#             new_row = pd.DataFrame([{'filename': filename,
#                                               'answer': answer,
#                                               'query_text': query,
#                                               'program_text': program}])
#             que2prog_df = pd.concat([que2prog_df, new_row], ignore_index=True)
    
#     # Save to csv files
#     img_det_df.to_csv(os.path.join(data_dir, f'{prefix}_img_det.csv'), index=False)
#     que2prog_df.to_csv(os.path.join(data_dir, f'{prefix}_q2p.csv'), index=False)
    
# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser(description='Sort-of-CLEVR NSAI Dataset Generator')
#     parser.add_argument('--n_train', type=int, default=50, help='number of train images to generates')
#     parser.add_argument('--n_test', type=int, default=1000, help='number of test images to generates')
#     args = parser.parse_args()
    
#     print('Building Train Dataset...')
#     build_dataset(args.n_train, 'train', 'train')
#     print('Building Test Dataset...')
#     build_dataset(args.n_test, 'test', 'test')