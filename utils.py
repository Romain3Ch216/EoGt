import numpy as np
import spectral
import cv2
import ast
import pandas as pd

def hyper2rgb(img, bands):
    """Convert hyperspectral cube to a rgb image.
    Args:
        img: HS npy cube
        bands: tuple of rgb bands
    Returns:
        npy rgb array
    """
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')
    return rgb

class GtSelection:
    def __init__(self, HS_img):
        self.HS_img = np.load(HS_img)
        self.rgb_bands = (50,30,15)
        self.rgb = hyper2rgb(self.HS_img,self.rgb_bands)
        self.rgb = self.rgb[:,:,::-1]
        self.RGB = cv2.UMat(self.rgb)
        self.image = cv2.UMat(np.copy(self.rgb))
        self.pt = None
        self.pts = []
        self.next = False
        self.reset = False
        set = {'labels': [], 'colors': [], 'nb_px': []}
        self.set = pd.DataFrame(set)
        self.n_bands = self.HS_img.shape[-1]

    def pick(self, event, x, y, flags, param):
        if event == 1:
            self.pts.append([x,y])
        if event == 0:
            self.pt = [x,y]

    def select_gt(self,n_class,colors):
        rgb = self.rgb
        IMG = self.HS_img
        gt  = np.zeros((rgb.shape[0],rgb.shape[1]))
        cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)
        cv2.setMouseCallback('image', self.pick)
        colors = [ast.literal_eval(color) for color in colors]
        colors = [tuple((color[2],color[1],color[0],color[3])) for color in colors]
        nb_px = []
        class_id = 1

        while class_id < n_class+1:

            # display the image and wait for a keypress
            image = cv2.imshow('image', self.RGB)
            key = cv2.waitKey(1) & 0xFF
            color = colors[class_id-1]

            if key == ord("f"):
                # Validate polygon
                polygon = np.array(self.pts)
                self.pts = []
                image = cv2.fillConvexPoly(self.RGB, polygon, color)
                cv2.fillConvexPoly(gt, polygon, class_id)

            elif self.next:
                nb_px.append(np.sum(gt==class_id))
                class_id += 1
                self.next = False
                self.image = self.RGB.get()

            elif self.reset:
                self.RGB = cv2.UMat(self.image)
                gt[gt==class_id] = 0
                self.reset = False

        cv2.destroyWindow('image')
        np.save('gt.npy',gt)
        cv2.imwrite('gt.jpg', self.RGB, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return nb_px
