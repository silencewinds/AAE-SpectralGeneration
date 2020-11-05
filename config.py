# net params
mb_size = 20         #batch_size
z_dim = 32

h_dim_0 = 1024
h_dim_1 = 512
h_dim_2 = 256
h_dim_3 = 128

# train params
lr = 1e-4
epoches = 50000
num_samples = 3233   #样本数量

# file
data_file_path = "m_data/"
data_type = "a0_40_50"
model_path = "models/" + data_type

m_data = data_file_path + data_type + ".mat"   #原始mat数据
np_data = "np_data/" + data_type + ".npy"      #处理后（包括归一化）的numpy数据，将其装入data_loader

out_img_path = "out_img/" + data_type
out_np_path = "out_np/" + data_type

predict_img_path = "predict_img/" + data_type
predict_np_path = "predict_np/" + data_type
