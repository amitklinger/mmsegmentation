# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
# import os.path as osp
import torch
# import torch.nn as nn

from pas_ss_pp import PostProcess
import onnx
from onnxsim import simplify


def parse_args():
    parser.add_argument('--no_simplify', action='store_false')
    parser.add_argument('--shape', nargs=2, type=int, default=[92, 120])
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('-o', '--opset', type=int, default=13)
    parser.add_argument('--out_name', default='fcn.onnx', type=str, help="Name for the onnx output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = PostProcess(args.num_classes)
    if device == 'cuda':
        model = model.cuda()
    model = model.eval()

    # (1, 10, 92, 120)
    imgs = torch.zeros(1, args.num_classes, args.shape[0], args.shape[1], dtype=torch.float32).to(device)

    torch.onnx.export(model,
                      imgs, args.out_name,
                      input_names=['test_input'],
                      output_names=['output'],
                      training=torch.onnx.TrainingMode.PRESERVE,
                      do_constant_folding=False,
                      opset_version=args.opset)

    # if also simplify
    if args.no_simplify:
        model_onnx = onnx.load(args.out_name)
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, args.out_name)
        print(f"Simplified model saved at: {args.out_name}")
    else:
        print(f"Model saved at: {args.out_name}")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        epilog='Example: python tools/export_pp_2onnx.py --num-classes 10 --shape 92 120 --out_name fcn_hailo_pp8.onnx -o 13')
    main()
