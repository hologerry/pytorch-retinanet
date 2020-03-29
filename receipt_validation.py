import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import ReceiptDataset, Resizer, Normalizer
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    dataset_val = ReceiptDataset(parser.coco_path, set_name='val',
                                 transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True)

    use_gpu = False

    retinanet = retinanet.to(device)

    retinanet.load_state_dict(torch.load(parser.model_path, map_location=device))
    retinanet.training = False
    retinanet.eval()
    retinanet.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()
