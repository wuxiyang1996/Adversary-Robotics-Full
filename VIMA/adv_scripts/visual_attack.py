import cv2
import numpy as np
import random

# visual_attack_cfg = {
#     "None": {},
#     "blurring": {"size": [11, 11]},
#     "noise": {"mean": 0, "std": 25},
#     "affine_trasforms": {"x_affine": [-0.2, 0.2], "y_affine": [-0.2, 0.2]},
#     "rotate_trasforms": {"rot_scope": [-90, 90]},
#     "cropping": {"x_min": [0, 0.2], "y_min": [0, 0.2], "x_max": [0.8, 1], "y_max": [0.8, 1]},
#     "addition_seg": {"x_pos": [0.2, 0.8], "y_pos": [0.2, 0.8], "x_size": [0.1, 0.3], "y_size": [0.1, 0.3]},
#     "addition_rgb": {"x_pos": [0.2, 0.8], "y_pos": [0.2, 0.8], "x_size": [0.1, 0.3], "y_size": [0.1, 0.3]}
# }

def add_gaussian_noise(image, mean=0, std=25):
    image = image.astype(np.uint8)
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

class visual_attack:
    def __init__(self, attack_approach, cfg, img, obj_id):
        self.attack_approach = attack_approach
        self.cfg = cfg[attack_approach]
        self.hyper_para = {}
        self.obj_id = obj_id
        C, W, H = img.shape
        self.C = C
        self.W = W
        self.H = H

        self.initialize(img)

    def initialize(self, img):
        C, W, H = img.shape
        if self.attack_approach == "affine_trasforms":
            self.hyper_para["x_affine"] = random.randint(int(self.cfg["x_affine"][0] * W), int(self.cfg["x_affine"][1] * W))
            self.hyper_para["y_affine"] = random.randint(int(self.cfg["y_affine"][0] * H), int(self.cfg["y_affine"][1] * H))
        elif self.attack_approach == "rotate_trasforms":
            self.hyper_para["rot"] = random.random() * (-self.cfg["rot_scope"][0] + self.cfg["rot_scope"][1]) + self.cfg["rot_scope"][0]
        elif self.attack_approach == "cropping":
            self.hyper_para["x_min"] = random.randint(int(self.cfg["x_min"][0] * W), int(self.cfg["x_min"][1] * W))
            self.hyper_para["x_max"] = random.randint(int(self.cfg["x_max"][0] * W), int(self.cfg["x_max"][1] * W))
            self.hyper_para["y_min"] = random.randint(int(self.cfg["y_min"][0] * H), int(self.cfg["y_min"][1] * H))
            self.hyper_para["y_max"] = random.randint(int(self.cfg["y_max"][0] * H), int(self.cfg["y_max"][1] * H))
        elif self.attack_approach == "addition_seg":
            self.hyper_para["x_pos"] = random.randint(int(self.cfg["x_pos"][0] * W), int(self.cfg["x_pos"][1] * W))
            self.hyper_para["x_size"] = random.randint(int(self.cfg["x_size"][0] * W), int(self.cfg["x_size"][1] * W))
            self.hyper_para["y_pos"] = random.randint(int(self.cfg["y_pos"][0] * H), int(self.cfg["y_pos"][1] * H))
            self.hyper_para["y_size"] = random.randint(int(self.cfg["y_size"][0] * H), int(self.cfg["y_size"][1] * H))
            self.hyper_para["obj_id"] = random.choice(self.obj_id)
        elif self.attack_approach == "addition_rgb":
            self.hyper_para["x_pos"] = random.randint(int(self.cfg["x_pos"][0] * W), int(self.cfg["x_pos"][1] * W))
            self.hyper_para["x_size"] = random.randint(int(self.cfg["x_size"][0] * W), int(self.cfg["x_size"][1] * W))
            self.hyper_para["y_pos"] = random.randint(int(self.cfg["y_pos"][0] * H), int(self.cfg["y_pos"][1] * H))
            self.hyper_para["y_size"] = random.randint(int(self.cfg["y_size"][0] * H), int(self.cfg["y_size"][1] * H))


    def implement_attack(self, rgb_img, seg_img):
        # rgb_img, seg_img = rgb_img.transpose((1, 2, 0)), seg_img
        if self.attack_approach == "None":
            return rgb_img, seg_img
        elif self.attack_approach == "blurring":
            return self.blurring_attack(rgb_img, seg_img)
        elif self.attack_approach == "noise":
            return self.noise_attack(rgb_img, seg_img)
        elif self.attack_approach == "affine_trasforms":
            return self.affine_trasforms_attack(rgb_img, seg_img)
        elif self.attack_approach == "rotate_trasforms":
            return self.rotate_trasforms_attack(rgb_img, seg_img)
        elif self.attack_approach == "cropping":
            return self.cropping_attack(rgb_img, seg_img)
        elif self.attack_approach == "addition_seg":
            return self.addition_seg_attack(rgb_img, seg_img)
        elif self.attack_approach == "addition_rgb":
            return self.addition_rgb_attack(rgb_img, seg_img)

    def blurring_attack(self, rgb_img, seg_img):
        rgb_img_out = cv2.GaussianBlur(rgb_img, self.cfg["size"], 0)
        # seg_img_out = cv2.GaussianBlur(seg_img, self.cfg["size"], 0)
        return rgb_img_out, seg_img

    def noise_attack(self, rgb_img, seg_img):
        rgb_img_out = add_gaussian_noise(rgb_img, self.cfg["mean"], self.cfg["std"])
        # seg_img_out = add_gaussian_noise(seg_img, self.cfg["mean"], self.cfg["std"])
        return rgb_img_out, seg_img

    def affine_trasforms_attack(self, rgb_img, seg_img):
        rgb_img = rgb_img.transpose(1, 2, 0)
        M = np.float32([[1, 0, self.hyper_para["x_affine"]],[0, 1, self.hyper_para["y_affine"]]])
        rgb_img_out = cv2.warpAffine(rgb_img, M, (self.H, self.W)).transpose(2, 0, 1)
        seg_img_out = cv2.warpAffine(seg_img, M, (self.H, self.W))
        return rgb_img_out, seg_img_out

    def rotate_trasforms_attack(self, rgb_img, seg_img):
        rgb_img = rgb_img.transpose(1, 2, 0)
        M = cv2.getRotationMatrix2D((self.W/2, self.H/2), self.hyper_para["rot"], 1) 
        rgb_img_out = cv2.warpAffine(rgb_img, M, (self.H, self.W)).transpose(2, 0, 1)
        seg_img_out = cv2.warpAffine(seg_img, M, (self.H, self.W))
        return rgb_img_out, seg_img_out

    def cropping_attack(self, rgb_img, seg_img):
        x_min, x_max = self.hyper_para["x_min"], self.hyper_para["x_max"]
        y_min, y_max = self.hyper_para["y_min"], self.hyper_para["y_max"]
        rgb_img_out = rgb_img[:, x_min: x_max + 1, y_min: y_max + 1].transpose(1, 2, 0)
        rgb_img_out = cv2.resize(rgb_img_out, (self.H, self.W)).transpose(2, 0, 1)
        seg_img_out = seg_img[x_min: x_max + 1, y_min: y_max + 1]
        seg_img_out = cv2.resize(seg_img_out, (self.H, self.W))
        return rgb_img_out, seg_img_out

    def addition_seg_attack(self, rgb_img, seg_img):
        x_pos, x_size = self.hyper_para["x_pos"], self.hyper_para["x_size"]
        y_pos, y_size = self.hyper_para["y_pos"], self.hyper_para["y_size"]

        x = np.arange(int(x_pos - x_size/2), int(x_pos + x_size/2))
        y = np.arange(int(y_pos - y_size/2), int(y_pos + y_size/2))
        xv, yv = np.meshgrid(x, y)

        seg_img[xv, yv] = self.hyper_para["obj_id"]
        return rgb_img, seg_img

    def addition_rgb_attack(self, rgb_img, seg_img):
        x_pos, x_size = self.hyper_para["x_pos"], self.hyper_para["x_size"]
        y_pos, y_size = self.hyper_para["y_pos"], self.hyper_para["y_size"]

        x = np.arange(int(x_pos - x_size/2), int(x_pos + x_size/2))
        y = np.arange(int(y_pos - y_size/2), int(y_pos + y_size/2))
        xv, yv = np.meshgrid(x, y)

        rgb_img[:, xv, yv] = 255
        return rgb_img, seg_img

# attack_approach_list = [
#     "None",
#     "blurring",
#     "noise",
#     "affine_trasforms",
#     "rotate_trasforms",
#     "cropping",
#     "addition_seg",
#     "addition_rgb"
# ]
# attack_approach = "affine_trasforms"
# img = np.zeros((3, 256, 128))
# seg = np.zeros((256, 128))
# obj_id = [5, 6, 7]
# vis = visual_attack(attack_approach, visual_attack_cfg, img, obj_id)
# rgb_img, seg_img = vis.implement_attack(img, seg)