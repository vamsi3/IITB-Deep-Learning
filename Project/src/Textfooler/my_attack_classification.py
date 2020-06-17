import os

command = 'python final3.py --dataset_path data/imdb ' \
          '--data_size 30 ' \
          '--target_model bert ' \
          '--target_model_path ./extras/imdb ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path ./extras/counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./extras/cos_sim_counter_fitting.npy ' \
          '--USE_cache_path ./tf_cache '

os.system(command)
