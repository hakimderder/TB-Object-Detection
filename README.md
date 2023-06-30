# TB-Object-Detection

Pour tous les programmes la lecture du flux de la caméra est fait par index avec cv.VideoCapture(2). L'index peut être différent. Tester avec nvgstcapture-1.0 --camsrc=0 --cap-dev-node=\<N>

#### YoloFastestV2

Version avec détection d'objets et mesure de la distance

Dans `yoloFastestV2-usb` : 

Charger le .cbp dans CodeBlocks

Compiler et lancer le programme (ça lance le mainFV2.cpp)

Pour entraîner le modèle, marche à suivre sur https://github.com/dog-qiuqiu/Yolo-FastestV2. Les fichiers nécessaires sont dans train/

#### Edge detection

Détection de contours avec sobel :

Lancer `python sobel.py`

Détection des cercles :

Lancer `python circles.py `

Détection de contours avec des fonctions d'OpenCV (marche pas vraiment) : 

Lancer `python contours.py `

#### Color

Tentative de séparation par luminance .Modifie l'image de RGB en YUV.

Charger le .cbp dans CodeBlocks

Compiler et lancer le programme (ça lance le mainFV2.cpp)