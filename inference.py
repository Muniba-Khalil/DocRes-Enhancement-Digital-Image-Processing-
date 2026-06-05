import os
import cv2
import glob
from pathlib import Path
import utils
import argparse
import numpy as np
import torch

from utils import convert_state_dict
from models import restormer_arch
from data.preprocess.crop_merge_image import stride_integral

os.sys.path.append('./data/MBD/')
from data.MBD.infer import net1_net2_infer_single_im

from PIL import Image, ImageEnhance


def dewarp_prompt(img):
    mask = net1_net2_infer_single_im(img, 'data/MBD/checkpoint/mbd.pkl')
    base_coord = utils.getBasecoord(256, 256) / 256
    img[mask == 0] = 0
    mask = cv2.resize(mask, (256, 256)) / 255
    return img, np.concatenate((base_coord, np.expand_dims(mask, -1)), -1)


def deshadow_prompt(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (1024, 1024))
    rgb_planes = cv2.split(img)
    bg_imgs = []

    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        bg_imgs.append(bg_img)

    bg_imgs = cv2.merge(bg_imgs)
    bg_imgs = cv2.resize(bg_imgs, (w, h))
    return bg_imgs


def deblur_prompt(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    high_frequency = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    high_frequency = cv2.cvtColor(high_frequency, cv2.COLOR_BGR2GRAY)
    high_frequency = cv2.cvtColor(high_frequency, cv2.COLOR_GRAY2BGR)
    return high_frequency


def appearance_prompt(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (1024, 1024))
    rgb_planes = cv2.split(img)
    result_norm_planes = []

    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm, (w, h))
    return result_norm


def binarization_promptv2(img):
    result, thresh = utils.SauvolaModBinarization(img)
    thresh = thresh.astype(np.uint8)
    result[result > 155] = 255
    result[result <= 155] = 0

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    high_frequency = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    high_frequency = cv2.cvtColor(high_frequency, cv2.COLOR_BGR2GRAY)

    return np.concatenate((np.expand_dims(thresh, -1),
                           np.expand_dims(high_frequency, -1),
                           np.expand_dims(result, -1)), -1)


def dewarping(model, im_path):
    INPUT_SIZE = 256
    im_org = cv2.imread(im_path)
    h0, w0 = im_org.shape[:2]

    if args.resize != 1.0:
        new_size = (int(w0 * args.resize), int(h0 * args.resize))
        im_org = cv2.resize(im_org, new_size)

    im_masked, prompt_org = dewarp_prompt(im_org.copy())
    h, w = im_masked.shape[:2]

    im_masked = cv2.resize(im_masked, (INPUT_SIZE, INPUT_SIZE))
    im_masked = im_masked / 255.0
    im_masked = torch.from_numpy(im_masked.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    prompt = torch.from_numpy(prompt_org.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    in_im = torch.cat((im_masked, prompt), dim=1)

    base_coord = utils.getBasecoord(INPUT_SIZE, INPUT_SIZE) / INPUT_SIZE
    model = model.float()

    with torch.no_grad():
        pred = model(in_im)
        pred = pred[0][:2].permute(1, 2, 0).cpu().numpy()
        pred = pred + base_coord

    for _ in range(15):
        pred = cv2.blur(pred, (3, 3), borderType=cv2.BORDER_REPLICATE)

    pred = cv2.resize(pred, (w, h)) * (w, h)
    pred = pred.astype(np.float32)
    out_im = cv2.remap(im_org, pred[:, :, 0], pred[:, :, 1], cv2.INTER_LINEAR)

    prompt_org = (prompt_org * 255).astype(np.uint8)
    prompt_org = cv2.resize(prompt_org, im_org.shape[:2][::-1])

    return prompt_org[:, :, 0], prompt_org[:, :, 1], prompt_org[:, :, 2], out_im


def appearance(model, im_path):
    MAX_SIZE = 1600
    im_org = cv2.imread(im_path)
    h0, w0 = im_org.shape[:2]

    if args.resize != 1.0:
        new_size = (int(w0 * args.resize), int(h0 * args.resize))
        im_org = cv2.resize(im_org, new_size)

    prompt = appearance_prompt(im_org)
    in_im = np.concatenate((im_org, prompt), -1)

    if max(in_im.shape[:2]) < MAX_SIZE:
        in_im, padding_h, padding_w = stride_integral(in_im, 8)
    else:
        in_im = cv2.resize(in_im, (MAX_SIZE, MAX_SIZE))

    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2, 0, 1)).unsqueeze(0).half().to(DEVICE)
    model = model.half()

    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred, 0, 1)[0].permute(1, 2, 0).cpu().numpy()
        pred = (pred * 255).astype(np.uint8)

        if max(in_im.shape[2:]) < MAX_SIZE:
            out_im = pred[padding_h:, padding_w:]
        else:
            pred[pred == 0] = 1
            shadow_map = cv2.resize(im_org, (MAX_SIZE, MAX_SIZE)).astype(float) / pred.astype(float)
            shadow_map = cv2.resize(shadow_map, (im_org.shape[1], im_org.shape[0]))
            shadow_map[shadow_map == 0] = 1e-5
            out_im = np.clip(im_org.astype(float) / shadow_map, 0, 255).astype(np.uint8)

    return prompt[:, :, 0], prompt[:, :, 1], prompt[:, :, 2], out_im


def deshadowing(model, im_path):
    MAX_SIZE = 1600
    im_org = cv2.imread(im_path)
    h0, w0 = im_org.shape[:2]

    if args.resize != 1.0:
        new_size = (int(w0 * args.resize), int(h0 * args.resize))
        im_org = cv2.resize(im_org, new_size)

    prompt = deshadow_prompt(im_org)
    in_im = np.concatenate((im_org, prompt), -1)

    if max(in_im.shape[:2]) < MAX_SIZE:
        in_im, padding_h, padding_w = stride_integral(in_im, 8)
    else:
        in_im = cv2.resize(in_im, (MAX_SIZE, MAX_SIZE))

    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2, 0, 1)).unsqueeze(0).half().to(DEVICE)
    model = model.half()

    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred, 0, 1)[0].permute(1, 2, 0).cpu().numpy()
        pred = (pred * 255).astype(np.uint8)

        if max(in_im.shape[:2]) < MAX_SIZE:
            out_im = pred[padding_h:, padding_w:]
        else:
            pred[pred == 0] = 1
            shadow_map = cv2.resize(im_org, (MAX_SIZE, MAX_SIZE)).astype(float) / pred.astype(float)
            shadow_map = cv2.resize(shadow_map, (im_org.shape[1], im_org.shape[0]))
            shadow_map[shadow_map == 0] = 1e-5
            out_im = np.clip(im_org.astype(float) / shadow_map, 0, 255).astype(np.uint8)

    return prompt[:, :, 0], prompt[:, :, 1], prompt[:, :, 2], out_im


def deblurring(model, im_path):
    im_org = cv2.imread(im_path)
    h0, w0 = im_org.shape[:2]

    if args.resize != 1.0:
        new_size = (int(w0 * args.resize), int(h0 * args.resize))
        im_org = cv2.resize(im_org, new_size)

    in_im, padding_h, padding_w = stride_integral(im_org, 8)
    prompt = deblur_prompt(in_im)

    in_im = np.concatenate((in_im, prompt), -1)
    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2, 0, 1)).unsqueeze(0).half().to(DEVICE)
    model = model.half()

    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred, 0, 1)[0].permute(1, 2, 0).cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        out_im = pred[padding_h:, padding_w:]

    return prompt[:, :, 0], prompt[:, :, 1], prompt[:, :, 2], out_im


def binarization(model, im_path):
    im_org = cv2.imread(im_path)
    h0, w0 = im_org.shape[:2]

    if args.resize != 1.0:
        new_size = (int(w0 * args.resize), int(h0 * args.resize))
        im_org = cv2.resize(im_org, new_size)

    im, padding_h, padding_w = stride_integral(im_org, 8)
    prompt = binarization_promptv2(im)
    in_im = np.concatenate((im, prompt), -1)

    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2, 0, 1)).unsqueeze(0).half().to(DEVICE)
    model = model.half()

    with torch.no_grad():
        pred = model(in_im)[:, :2, :, :]
        pred = torch.max(torch.softmax(pred, 1), 1)[1][0].cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.resize(pred, (im.shape[1], im.shape[0]))
        out_im = pred[padding_h:, padding_w:]

    return prompt[:, :, 0], prompt[:, :, 1], prompt[:, :, 2], out_im


def get_args():
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--model_path', type=str, default='./checkpoints/docres.pkl')
    parser.add_argument('--im_path', type=str, default='./distorted/')
    parser.add_argument('--out_folder', type=str, default='./restorted/')
    parser.add_argument('--task', type=str, default='dewarping')
    parser.add_argument('--save_dtsprompt', type=int, default=0)

    parser.add_argument('--resize', type=float, default=1.0, help="Resize factor before inference")

    # Post processing
    parser.add_argument('--contrast', type=float, default=1.0, help="Enhance contrast after restoration")
    parser.add_argument('--sharpness', type=float, default=1.0, help="Enhance sharpness after restoration")
    parser.add_argument('--denoise', type=int, default=0, help="Apply denoise postprocessing (0/1)")
    parser.add_argument('--clahe', type=int, default=0, help="Apply CLAHE postprocessing (0/1)")

    # Strong visible enhancement
    parser.add_argument('--gamma', type=float, default=1.0, help="Gamma correction (>1 brightens)")
    parser.add_argument('--saturation', type=float, default=1.0, help="Saturation boost (>1 more vivid)")

    # Output naming
    parser.add_argument('--tag', type=str, default="", help="Short tag added to output filename")

    args = parser.parse_args()
    possible_tasks = ['dewarping', 'deshadowing', 'appearance', 'deblurring', 'binarization', 'end2end', 'enhance_only']
    assert args.task in possible_tasks, 'Unsupported task, task must be one of ' + ', '.join(possible_tasks)
    return args


def model_init(args):
    model = restormer_arch.Restormer(
        inp_channels=6,
        out_channels=3,
        dim=48,
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=True
    )

    if DEVICE.type == 'cpu':
        state = convert_state_dict(torch.load(args.model_path, map_location='cpu')['model_state'])
    else:
        state = convert_state_dict(torch.load(args.model_path, map_location='cuda:0')['model_state'])

    model.load_state_dict(state)
    model.eval()
    model = model.to(DEVICE)
    return model


def inference_one_im(model, im_path, task):

    # NEW FAST MODE: Enhance only (no model inference)
    if task == "enhance_only":
        im_org = cv2.imread(im_path)
        if im_org is None:
            raise ValueError(f"Cannot read image: {im_path}")
        # Dummy prompt outputs, enhancement applied in save_results()
        return im_org[:, :, 0], im_org[:, :, 0], im_org[:, :, 0], im_org

    if task == 'dewarping':
        prompt1, prompt2, prompt3, restorted = dewarping(model, im_path)
    elif task == 'deshadowing':
        prompt1, prompt2, prompt3, restorted = deshadowing(model, im_path)
    elif task == 'appearance':
        prompt1, prompt2, prompt3, restorted = appearance(model, im_path)
    elif task == 'deblurring':
        prompt1, prompt2, prompt3, restorted = deblurring(model, im_path)
    elif task == 'binarization':
        prompt1, prompt2, prompt3, restorted = binarization(model, im_path)
    elif task == 'end2end':
        prompt1, prompt2, prompt3, restorted = dewarping(model, im_path)
        cv2.imwrite('restorted/step1.jpg', restorted)
        prompt1, prompt2, prompt3, restorted = deshadowing(model, 'restorted/step1.jpg')
        cv2.imwrite('restorted/step2.jpg', restorted)
        prompt1, prompt2, prompt3, restorted = appearance(model, 'restorted/step2.jpg')

    return prompt1, prompt2, prompt3, restorted


def apply_gamma(img_bgr, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)


def apply_postprocessing(img):
    out = img.copy()

    if args.clahe == 1:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    if args.saturation != 1.0:
        img_pil = ImageEnhance.Color(img_pil).enhance(args.saturation)

    if args.contrast != 1.0:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(args.contrast)

    if args.sharpness != 1.0:
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(args.sharpness)

    out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    if args.gamma != 1.0:
        out = apply_gamma(out, args.gamma)

    if args.denoise == 1:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)

    return out


def save_results(img_path, out_folder, task, save_dtsprompt):
    global restorted, prompt1, prompt2, prompt3

    im_name = os.path.split(img_path)[-1]
    im_format = '.' + im_name.split('.')[-1]

    tag = f"_{args.tag}" if args.tag != "" else ""
    save_path = os.path.join(out_folder, im_name.replace(im_format, f"_{task}{tag}{im_format}"))

    restorted_final = apply_postprocessing(restorted)
    cv2.imwrite(save_path, restorted_final)

    if save_dtsprompt:
        cv2.imwrite(save_path.replace(im_format, '_prompt1' + im_format), prompt1)
        cv2.imwrite(save_path.replace(im_format, '_prompt2' + im_format), prompt2)
        cv2.imwrite(save_path.replace(im_format, '_prompt3' + im_format), prompt3)


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    model = model_init(args)

    img_source = args.im_path
    os.makedirs(args.out_folder, exist_ok=True)

    if Path(img_source).is_dir():
        img_paths = glob.glob(os.path.join(img_source, '*'))
        for img_path in img_paths:
            prompt1, prompt2, prompt3, restorted = inference_one_im(model, img_path, args.task)
            save_results(img_path, args.out_folder, args.task, args.save_dtsprompt)
    else:
        prompt1, prompt2, prompt3, restorted = inference_one_im(model, img_source, args.task)
        save_results(img_source, args.out_folder, args.task, args.save_dtsprompt)
