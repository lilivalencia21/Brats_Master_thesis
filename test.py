import torch
import numpy as np
import nibabel as nib
from functions.utilities import *
from functions.instructions import *
from functions.patches import *
from functions.nets import *

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = UNet3D()
# model.load_state_dict(torch.load("/home/liliana/models/net_norm_overlap.pth"))
# model.to(device)
# model.eval()

#To use with test_cross_validation function.
crossvalidation_cfg = {'model': UNet3D(),
                       'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       'model_path': '/home/liliana/models/crossvalidation/validationData_from_5_to_9_fold_2.pt',
                       'training_set_txt':"/home/liliana/dataToValidate/cases_train_fold_2.txt",
                       'path_to_save_segm': "/home/liliana/Results/crossvalidation_img/Fold_2",
                       'path_to_save_txt': "/home/liliana/Results/crossvalidation_img/Fold_2/"+ 'fold_2.txt'}

crossval_optim_cfg = {'model': UNet3D(),
                       'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       'model_path': "/home/liliana/models/crossvalidation_opt/validationData_from_0_to_4_fold_1.pt",
                       'training_set_txt':"/home/liliana/dataToValidate/optimizing/cases_train_fold_1.txt",
                       'path_to_save_segm': "/home/liliana/Results/crossval_optim/Fold_1/",
                       'path_to_save_txt': "/home/liliana/Results/crossval_optim/Fold_1/"+ 'fold_1.txt'}



data_dir_test = "/home/liliana/Data/train"
dataset_test = load_dataset(data_dir_test)

test_cross_validation(dataset_test, crossval_optim_cfg)



# # data_dir_test = os.path.expanduser("~/Desktop/Liliana/Data/valid")
# data_dir_test = "/home/liliana/Data/valid/"
# dataset_test = load_dataset(data_dir_test)
# gt = load_images(dataset_test[0]['gt_path'])
#
# nifti_image = nib.load(dataset_test[0]['image_paths'][0])
# original_image = load_images(dataset_test[0]['image_paths']).astype(np.float)
# mean = dataset_test[0]['mean']
# std_dev = dataset_test[0]['std_dev']
# input_image = np.stack([norm(image, mean, std) for image, mean, std in zip(original_image, mean, std_dev)])
#
#
# img = np.expand_dims(input_image, axis=0)
# test_input = torch.tensor(img, dtype=torch.float32, requires_grad=False, device=device)
#
# with torch.no_grad():
#     testing = model(test_input)
#
# testing_np = testing.cpu().detach().numpy()
#
# # nclasses = 5
# results = np.argmax(testing_np, axis=1)
# result_prob = np.squeeze(results, axis=0)
#
#
# dice = dice_multiclass(gt, result_prob)
# print(dice)

#Save Results
# result_img = nib.Nifti1Image(result_prob, nifti_image.affine, nifti_image.header)
# image_filepath = os.path.join("/home/liliana/Results/Batch12/", 'test.nii.gz')
# print("Saving {}...".format('test.nii.gz'))
# nib.save(result_img, image_filepath )


# result_probs = [result_prob]
# for i, prob in enumerate(result_probs):
#     for j in range(nclasses):
#         result_img = nib.Nifti1Image(prob[j], nifti_image.affine, nifti_image.header)
#         image_filepath = os.path.join("/home/liliana/Results/Batch12/", 'test{}_class{}.nii.gz'.format(i, j))
#         print("Saving {}...".format('test{}_class{}.nii.gz'.format(i, j)))
#         nib.save(result_img, image_filepath )
