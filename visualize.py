import argparse
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet.dataloader import (AspectRatioBasedSampler,
                                  CocoDataset, CSVDataset, Normalizer,
                                  ReceiptDataset, Resizer, UnNormalizer,
                                  collater)

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='receipt', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', default='data/receipt_0158_0062', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)
    if parser.dataset == 'receipt':
        dataset_val = ReceiptDataset(parser.coco_path, set_name='val',
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='train2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError(
            'Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(
        dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(
        dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = torch.load(parser.model)

    use_gpu = True

    retinanet = retinanet.to(device)

    retinanet = torch.nn.DataParallel(retinanet).to(device)

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(
                    data['img'].to(device).float())
            else:
                scores, classification, transformed_anchors = retinanet(
                    data['img'].float())
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu() > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(
                    classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2),
                              color=(0, 0, 255), thickness=2)
                print(label_name)

            cv2.imshow('img', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
