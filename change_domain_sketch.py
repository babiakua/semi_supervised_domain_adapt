import torch.cuda

from diffusionclip import DiffusionCLIP
from main import dict2namespace
import yaml
import os
import torchvision
import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.path.join('checkpoint', 'human_sketch_t601.pth')
align_face = True
edit_type = 'Sketch'
degree_of_change = 1


def change_domain(path_to_images, path_to_save, model_path, mode='train'):
    base_path = os.path.join(path_to_images, mode)
    image_paths = os.listdir(base_path)
    image_paths = [os.path.join(base_path, k) for k in image_paths]
    t_0 = int(model_path.split('_t')[-1].replace('.pth', ''))

    for image_path in tqdm.tqdm(image_paths):
        exp_dir = f"runs/MANI_{image_path.split('/')[-1]}_align{align_face}_mode{mode}"
        os.makedirs(exp_dir, exist_ok=True)
        n_inv_step = 40
        n_test_step = 6
        args_dic = {
            'config': 'celeba.yml',
            't_0': t_0,
            'n_inv_step': int(n_inv_step),
            'n_test_step': int(n_test_step),
            'sample_type': 'ddim',
            'eta': 0.0,
            'bs_test': 1,
            'model_path': model_path,
            'img_path': image_path,
            'deterministic_inv': 1,
            'hybrid_noise': 0,
            'n_iter': 1,
            'align_face': align_face,
            'image_folder': exp_dir,
            'model_ratio': degree_of_change,
            'edit_attr': None, 'src_txts': None, 'trg_txts': None,
        }
        args = dict2namespace(args_dic)
        with open(os.path.join('configs', args.config), 'r') as f:
            config_dic = yaml.safe_load(f)
        config = dict2namespace(config_dic)
        config.device = device
        runner = DiffusionCLIP(args, config)
        generated = runner.edit_one_image()
        img_ = image_path[-9:]

        torchvision.utils.save_image(generated, os.path.join(path_to_save, mode, img_))


if __name__ == "__main__":
    change_domain('../celeba_domain_base', '../celeba_domain_train_and_test_sketched', 'checkpoint/human_sketch_t601.pth', mode='train')



