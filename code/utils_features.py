# utils_features.py
import cv2, numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

TARGET_SIZE = (640,640)
DEEP_DIM = 2048

try:
    BASE_MODEL = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                          input_shape=(640,640,3))
    print("ResNet cargada.")
except Exception as e:
    print("No se pudo cargar ResNet:", e)
    BASE_MODEL = None

def preprocess_image(path):
    img = cv2.imread(str(path))
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    return img.astype("float32")/255.0

def extract_color(img):
    u = (img*255).astype(np.uint8)
    feats=[]
    for i in range(3):
        h=cv2.calcHist([u],[i],None,[64],[0,256]); cv2.normalize(h,h)
        feats+=h.flatten().tolist()
    hsv=cv2.cvtColor(u,cv2.COLOR_RGB2HSV)
    for i in range(2):
        h=cv2.calcHist([hsv],[i],None,[64],[0,180 if i==0 else 256]); cv2.normalize(h,h)
        feats+=h.flatten().tolist()
    return np.array(feats)

def extract_shape(img):
    g=cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(g,100,200)
    canny=np.sum(edges)/(640*640)
    hogf=hog(g,9,(32,32),(2,2),transform_sqrt=True)
    return np.concatenate([[canny],hogf])

def extract_texture(img):
    g=cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    lbp=local_binary_pattern(g,8,1,"uniform")
    h,_=np.histogram(lbp.ravel(),bins=10,range=(0,10))
    h=h/(h.sum()+1e-7)
    levels=32
    gq=(g//(256//levels)).astype(np.uint8)
    glcm=graycomatrix(gq,[1],[0],levels=levels,normed=True,symmetric=True)
    props=[graycoprops(glcm,p)[0,0] for p in ["contrast","energy","homogeneity","correlation"]]
    return np.concatenate([h,props])

def extract_deep(img):
    if BASE_MODEL is None:
        return np.zeros(DEEP_DIM, dtype=np.float32)
    b=np.expand_dims(img*255,0)
    b=preprocess_input(b)
    f=BASE_MODEL.predict(b,verbose=0)
    return f.flatten()
