import os
import json

import cv2
import numpy as np
import torchvision.transforms as tf
from PIL import Image
from torch.utils import data
from shapely.geometry import Polygon


class CustomDataset(data.Dataset):
    def __init__(self, config, phase='test', files='example/run/line/'):
        self.config = config
        self.phase = phase
        self.max_objs = config.max_objs
        self.transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.K = np.array([[762, 0, 640], [0, -762, 360], [0, 0, 1]],
                          dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)

        self.files = files
        self.filenames = [f for f in os.listdir(files) if f.endswith('.jpg')]
        #print("file", self.filenames)

    def padimage(self, image):
        outsize = [384, 640, 3]
        h, w = image.shape[0], image.shape[1]
        padimage = np.zeros(outsize, dtype=np.uint8)
        padimage[:h, :w] = image
        return padimage, outsize[0], outsize[1]

    def __getitem__(self, index):

        layout_name = os.path.join(self.files, self.filenames[index].rstrip('.jpg') + '_layout.json')

        print("file : ",self.files + self.filenames[index])
        img = Image.open(self.files + self.filenames[index])
        #img = img.resize((1280, 720))
        inh, inw = self.config.input_h, self.config.input_w
        orih, oriw = img.size[1], img.size[0]
        ratio_w = oriw / inw
        ratio_h = orih / inh
        #assert ratio_h == ratio_w == 2
        img = np.array(img)[:, :, [0, 1, 2]]
        img = cv2.resize(img, (inw, inh), interpolation=cv2.INTER_LINEAR)
        img, inh, inw = self.padimage(img)
        img = self.transforms(img)
        ret = {'img': img}
        ret['intri'] = self.K
        ret['intri_inv'] = self.K_inv

        #ret['filename']=self.filenames[index]

        pparams, labels, segs = self.dataload(
            layout_name, ratio_h, ratio_w,inh, inw)


        #add filename
        #ret["file"]=self.files + self.filenames[index]

        oh, ow = inh // self.config.downsample, inw // self.config.downsample
        x = np.arange(ow * 8)
        y = np.arange(oh * 8)
        xx, yy = np.meshgrid(x, y)
        xymap = np.stack([xx, yy], axis=2).astype(np.float32)
        oxymap = cv2.resize(xymap, (ow, oh), interpolation=cv2.INTER_LINEAR)
        oxy1map = np.concatenate([
            oxymap, np.ones_like(oxymap[:, :, :1])], axis=-1).astype(np.float32)
        ret['oxy1map'] = oxy1map
        
        # evaluate gt
        oseg = cv2.resize(segs, (ow, oh), interpolation=cv2.INTER_NEAREST)
        ret['iseg'] = segs
        ret['oseg'] = oseg

        #ret['iseg'] = np.ones([inh, inw])
        #print("oseg", oseg, segs)

        ixymap = cv2.resize(xymap, (inw, inh), interpolation=cv2.INTER_LINEAR)
        ixy1map = np.concatenate([
            ixymap, np.ones_like(ixymap[:, :, :1])], axis=-1).astype(np.float32)
        ret['ixy1map'] = ixy1map
        #ret['iseg'] = np.ones([inh, inw])
        ret['ilbox'] = np.zeros(20)
        return ret

    def dataload(self, layout_name, ratio_h,ratio_w, inh, inw):
        # planes
        with open(layout_name, 'r') as f:
            anno_layout = json.load(f)
            junctions = anno_layout['junctions']
            planes = anno_layout['planes']
            #print("loading", junctions, planes)

            coordinates = []
            for k in junctions:
                coordinates.append(k['coordinate'])

            coordinates = np.array(coordinates)
            orih, oriw = coordinates[:,1].max(), coordinates[:,0].max()
            # print("ori", oriw, orih)

            coordinates[:, 0] = coordinates[:, 0] / ratio_w  # x座標のスケーリング
            coordinates[:, 1] = coordinates[:, 1] / ratio_h  # y座標のスケーリング

            # coordinates[:, 0] = coordinates[:, 0]/ ratio_w  # x座標のスケーリング
            # coordinates[:, 1] = coordinates[:, 1] / ratio_h  # y座標のスケーリング


            pparams = []
            labels = []
            segs = -1 * np.ones([inh, inw])
            i = 0
            for pp in planes:
                if len(pp['visible_mask']) != 0:
                    if pp['type'] == 'wall':
                        cout = coordinates[pp['visible_mask'][0]]
                        polygon = Polygon(cout)
                        if polygon.area >= 1000:
                            cout = cout.astype(np.int32)
                            cv2.fillPoly(segs, [cout], color=i)
                           # pparams.append([*pp['normal'], pp['offset'] / 1000.])
                            #pparams.append([*(int(pp['normal'])), int(pp['offset']) ])
                            labels.append(0)
                            i = i + 1
                    else:
                        for v in pp['visible_mask']:
                            cout = coordinates[v]
                            polygon = Polygon(cout)
                            if polygon.area > 1000:
                                cout = cout.astype(np.int32)
                                cv2.fillPoly(segs, [cout], color=i)
                                #pparams.append([*(int(pp['normal'])), int(pp['offset']) ])
                                if pp['type'] == 'floor':
                                    labels.append(1)
                                else:
                                    labels.append(2)
                                i = i + 1
        #print("seg", len(segs), len(segs[0]))
        return pparams, labels, segs  

    def __len__(self):
        return len(self.filenames)
