import cv2
import numpy as np
import random


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
        if self.attack_approach == "affine_transforms":
            self.hyper_para["x_affine"] = random.randint(int(self.cfg["x_affine"][0] * W), int(self.cfg["x_affine"][1] * W))
            self.hyper_para["y_affine"] = random.randint(int(self.cfg["y_affine"][0] * H), int(self.cfg["y_affine"][1] * H))
        elif self.attack_approach == "rotate_transforms":
            self.hyper_para["rot"] = random.random() * (self.cfg["rot_scope"][1] - self.cfg["rot_scope"][0]) + self.cfg["rot_scope"][0]
        elif self.attack_approach == "cropping":
            self.hyper_para["x_min"] = random.randint(int(self.cfg["x_min"][0] * W), int(self.cfg["x_min"][1] * W))
            self.hyper_para["x_max"] = random.randint(int(self.cfg["x_max"][0] * W), int(self.cfg["x_max"][1] * W))
            self.hyper_para["y_min"] = random.randint(int(self.cfg["y_min"][0] * H), int(self.cfg["y_min"][1] * H))
            self.hyper_para["y_max"] = random.randint(int(self.cfg["y_max"][0] * H), int(self.cfg["y_max"][1] * H))
        elif self.attack_approach == "distortion":
            self.hyper_para["p1_x"] = random.randint(int(self.cfg["x_min"][0] * W), int(self.cfg["x_min"][1] * W))
            self.hyper_para["p1_y"] = random.randint(int(self.cfg["y_min"][0] * H), int(self.cfg["y_min"][1] * H))

            self.hyper_para["p2_x"] = random.randint(int(self.cfg["x_max"][0] * W), int(self.cfg["x_max"][1] * W))
            self.hyper_para["p2_y"] = random.randint(int(self.cfg["y_min"][0] * H), int(self.cfg["y_min"][1] * H))

            self.hyper_para["p3_x"] = random.randint(int(self.cfg["x_max"][0] * W), int(self.cfg["x_max"][1] * W))
            self.hyper_para["p3_y"] = random.randint(int(self.cfg["y_max"][0] * H), int(self.cfg["y_max"][1] * H))

            self.hyper_para["p4_x"] = random.randint(int(self.cfg["x_min"][0] * W), int(self.cfg["x_min"][1] * W))
            self.hyper_para["p4_y"] = random.randint(int(self.cfg["y_max"][0] * H), int(self.cfg["y_max"][1] * H))
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
        elif self.attack_approach == "filter":
            self.hyper_para["rgb"] = random.randint(0, 2)


    def implement_attack(self, rgb_img, seg_img):
        # rgb_img, seg_img = rgb_img.transpose((1, 2, 0)), seg_img
        if self.attack_approach == "None":
            return rgb_img, seg_img
        elif self.attack_approach == "blurring":
            return self.blurring_attack(rgb_img, seg_img)
        elif self.attack_approach == "noise":
            return self.noise_attack(rgb_img, seg_img)
        elif self.attack_approach == "filter":
            return self.filter_attack(rgb_img, seg_img)
        elif self.attack_approach == "affine_transforms":
            return self.affine_transforms_attack(rgb_img, seg_img)
        elif self.attack_approach == "rotate_transforms":
            return self.rotate_transforms_attack(rgb_img, seg_img)
        elif self.attack_approach == "cropping":
            return self.cropping_attack(rgb_img, seg_img)
        elif self.attack_approach == "distortion":
            return self.distortion_attack(rgb_img, seg_img)
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

    def filter_attack(self, rgb_img, seg_img):
        rgb_img[self.hyper_para["rgb"], :, :] = 255
        # seg_img_out = add_gaussian_noise(seg_img, self.cfg["mean"], self.cfg["std"])
        return rgb_img, seg_img

    def affine_transforms_attack(self, rgb_img, seg_img):
        rgb_img = rgb_img.transpose(1, 2, 0)
        # print(self.hyper_para["x_affine"], self.hyper_para["y_affine"])
        M = np.float32([[1, 0, self.hyper_para["x_affine"]],[0, 1, self.hyper_para["y_affine"]]])
        rgb_img_out = cv2.warpAffine(rgb_img, M, (self.H, self.W)).transpose(2, 0, 1)
        seg_img_out = cv2.warpAffine(seg_img, M, (self.H, self.W))
        return rgb_img_out, seg_img_out

    def rotate_transforms_attack(self, rgb_img, seg_img):
        rgb_img = rgb_img.transpose(1, 2, 0)
        M = cv2.getRotationMatrix2D((self.W/2, self.H/2), self.hyper_para["rot"], 1)
        # cv2.imwrite("test_1.jpg", rgb_img)
        rgb_img_out = cv2.warpAffine(rgb_img, M, (self.H, self.W)).transpose(2, 0, 1)
        # cv2.imwrite("test.jpg", rgb_img_out.transpose(1, 2, 0))
        seg_img_out = cv2.warpAffine(seg_img, M, (self.H, self.W))
        return rgb_img_out, seg_img_out

    def cropping_attack(self, rgb_img, seg_img):
        x_min, x_max = self.hyper_para["x_min"], self.hyper_para["x_max"]
        y_min, y_max = self.hyper_para["y_min"], self.hyper_para["y_max"]
        # print(self.hyper_para["x_min"], self.hyper_para["x_max"])
        rgb_img_out = rgb_img[:, x_min: x_max + 1, y_min: y_max + 1].transpose(1, 2, 0)
        # cv2.imwrite("test_1.jpg", rgb_img_out)
        rgb_img_out = cv2.resize(rgb_img_out, (self.H, self.W)).transpose(2, 0, 1)
        # cv2.imwrite("test.jpg", rgb_img_out.transpose(1, 2, 0))
        seg_img_out = seg_img[x_min: x_max + 1, y_min: y_max + 1]
        seg_img_out = cv2.resize(seg_img_out, (self.H, self.W))
        return rgb_img_out, seg_img_out

    def distortion_attack(self, rgb_img, seg_img):
        rgb_img = rgb_img.transpose(1, 2, 0)
        pt_A, pt_B, pt_C, pt_D = [[self.hyper_para["p1_y"], self.hyper_para["p1_x"]],
                                [self.hyper_para["p2_y"], self.hyper_para["p2_x"]],
                                [self.hyper_para["p3_y"], self.hyper_para["p3_x"]],
                                [self.hyper_para["p4_y"], self.hyper_para["p4_x"]]]
        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0], [0, self.W - 1],
                                [self.H - 1, self.W - 1], [self.H - 1, 0]])
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        # cv2.imwrite("test_1.jpg", rgb_img)
        rgb_img_out = cv2.warpPerspective(rgb_img, M, (self.H, self.W), flags=cv2.INTER_LINEAR).transpose(2, 0, 1)
        # cv2.imwrite("test.jpg", rgb_img_out.transpose(1, 2, 0))
        seg_img_out = cv2.warpPerspective(seg_img, M, (self.H, self.W), flags=cv2.INTER_LINEAR)
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