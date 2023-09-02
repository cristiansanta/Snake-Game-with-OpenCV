import cv2
import numpy as np
from time import time
import random
import math


video = cv2.VideoCapture(0)
#inicializando fuente para poner texto
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

#cargando la imagen de la manzana y haciendo que su máscara se superponga en el video
manzana = cv2.imread("M.png",-1)
manzana_mask = manzana[:,:,3]
manzana_mask_inv = cv2.bitwise_not(manzana_mask)
manzana = manzana[:,:,0:3]

# redimensionar imágenes de manzana
manzana = cv2.resize(manzana,(40,40),interpolation=cv2.INTER_AREA)
manzana_mask = cv2.resize(manzana_mask,(40,40),interpolation=cv2.INTER_AREA)
manzana_mask_inv = cv2.resize(manzana_mask_inv,(40,40),interpolation=cv2.INTER_AREA)
#iniciar una imagen en blanco y negro
blank_img = np.zeros((480,640,3),np.uint8)
#captura de video de la cámara web


#kernels para operaciones morfológicas
kernel_open = np.ones((4,4),np.uint8)
kernel_dilate = np.ones((15,15),np.uint8)

#para azul [99,115,150] [110,255,255]
#funcion para detectar color rojo|
def detect_red(hsv):
    #límite inferior para el valor de saturación de tono de color rojo
    color_low = np.array([49,50,50])  # 136,87,111
    color_up = np.array([107, 255, 255])  # 180,255,255
    mask = cv2.inRange(hsv, color_low, color_up)
    color_low2 = np.array([49,50,50])
    color_up2 = np.array([107, 255, 255])
    mask2 = cv2.inRange(hsv, color_low2, color_up2)
    maskred = mask+mask2
    
    cv2.imshow('final', maskred);
    maskred = cv2.erode(maskred, kernel_open, iterations=1)
    maskred = cv2.morphologyEx(maskred,cv2.MORPH_CLOSE,kernel_dilate)
    
    return maskred
   
#funciones para detectar la intersección de segmentos de línea.
def orientation(p,q,r):
    val = int(((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))
    if val == 0:
        #linear
        return 0
    elif (val>0):
        #sentido horario
        return 1
    else:
        #anti-horario
        return 2
def intersect(p,q,r,s):
    o1 = orientation(p, q, r)
    o2 = orientation(p, q, s)
    o3 = orientation(r, s, p)
    o4 = orientation(r, s, q)
    if(o1 != o2 and o3 != o4):
        return True

    return False

#tiempo de inicialización (usado para aumentar la longitud de la serpiente por segundo)
start_time = int(time())
# q utilizado para la inicialización de puntos
q,snake_len,score,temp=0,200,0,1
# almacena el punto central de la burbuja roja
point_x,point_y = 0,0
# almacena los puntos que satisfacen la condición, dist almacena dist entre 2 pts consecutivos, la longitud es len de serpiente
last_point_x,last_point_y,dist,length = 0,0,0,0
# almacena todos los puntos del cuerpo de la serpiente
points = []
# almacena la longitud entre todos los puntos
list_len = []
# generar un número aleatorio para la colocación de la imagen de la manzana
random_x = random.randint(10,550)
random_y = random.randint(10,400)
#utilizado para verificar intersecciones
a,b,c,d = [],[],[],[]
#main loop
while 1:
    xr, yr, wr, hr = 0, 0, 0, 0
    _,frame = video.read()
    # voltear el marco horizontalmente.
    frame = cv2.flip(frame,1)
    
    #Desenfoque el marco con el filtro gaussiano del tamaño de núcleo 11, para eliminar el ruido excesivo
    #El filtro gaussiano es un filtro de paso bajo que elimina los componentes de alta frecuencia se reducen
    blurred_frame = cv2.GaussianBlur(frame, (11,11), 0)
    
    # inicializando los puntos aceptados para que no estén en la esquina superior izquierda
    if(q==0 and point_x!=0 and point_y!=0):
        last_point_x = point_x
        last_point_y = point_y
        q=1
    #convirtiendo a hsv
    hsv = cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
    maskred = detect_red(hsv)
    # encontrar contornos
    contour_red, _ = cv2.findContours(maskred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    # dibujar un rectángulo alrededor del blob aceptado
    try:
        for i in range (0,10):
            xr, yr, wr, hr = cv2.boundingRect(contour_red[i])
            if (wr*hr)>2000:
                break
    except:
        pass
    cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 255, 255), 3)
    
    #haciendo cuerpo de serpiente
    point_x = int(xr+(wr/2))
    point_y = int(yr+(hr/2))
    # encontrar la distancia entre el último punto y el punto actual
    dist = int(math.sqrt(pow((last_point_x - point_x), 2) + pow((last_point_y - point_y), 2)))
    if (point_x!=0 and point_y!=0 and dist>5):
        #si se acepta el punto, se agrega a la lista de puntos y su longitud se agrega a list_len
        list_len.append(dist)
        length += dist
        last_point_x = point_x
        last_point_y = point_y
        points.append([point_x, point_y])
    #Si la longitud se vuelve mayor que la longitud esperada, eliminando puntos de la parte posterior para disminuir la longitud
    if (length>=snake_len):
        for i in range(len(list_len)):
            length -= list_len[0]
            list_len.pop(0)
            points.pop(0)
            if(length<=snake_len):
                break
    #inicializando imagen negra en blanco
    blank_img = np.zeros((480, 640, 3), np.uint8)
    # dibujando las líneas entre todos los puntos
    for i,j in enumerate(points):
        if (i==0):
            continue
        espesor = int(np.sqrt( len(list_len)  / float(i + 1)) * 4.5)
        cv2.line(blank_img, (points[i-1][0], points[i-1][1]), (j[0], j[1]), (0, 0, 255), espesor)
        cv2.circle(blank_img, (last_point_x, last_point_y), 3 , (0, 255, 0), 6)
    #si la serpiente come manzana, aumente la puntuación y encuentre una nueva posición para la manzana
    if  (last_point_x>random_x and last_point_x<(random_x+40) and last_point_y>random_y and last_point_y<(random_y+40)):
        score +=1
        random_x = random.randint(10, 550)
        random_y = random.randint(10, 400)
    #Agregar imagen en blanco para capturar fotograma
    frame = cv2.add(frame,blank_img)
    #Agregar imagen de manzana al marco
    roi = frame[random_y:random_y+40, random_x:random_x+40]
    img_bg = cv2.bitwise_and(roi, roi, mask=manzana_mask_inv)
    img_fg = cv2.bitwise_and(manzana, manzana, mask=manzana_mask)
    dst = cv2.add(img_bg, img_fg)
    frame[random_y:random_y + 40, random_x:random_x + 40] = dst
    
    
    # buscando serpientes golpeándose
    if(len(points)>5):
        # ayb son los puntos principales de la serpiente yc, d son todos los otros puntos
        b = points[len(points)-2]
        a = points[len(points)-1]
        for i in range(len(points)-3):
            c = points[i]
            d = points[i+1]
            if(intersect(a,b,c,d) and len(c)!=0 and len(d)!=0):
                temp = 0
                break
        if temp==0:
            break
    cv2.putText(frame, str("Puntaje = "+str(score)), (460, 20), font, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("frame",frame)
    # aumentando la longitud de la serpiente 40px por segundo
    if((int(time())-start_time)>1):
        snake_len += 40
        start_time = int(time())
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
cv2.putText(frame, str("Puntaje = "+str(score)), (460, 20), font, 1, (9,255,90), 2, cv2.LINE_AA)
cv2.imshow('final', maskred);
cv2.putText(frame, str("Perdiste!"), (160, 230), font, 3, (0, 0, 255), 3, cv2.LINE_AA)
cv2.putText(frame, str("Presione cualquier tecla pa salir."), (110, 290), font, 1, (255, 150, 0), 2, cv2.LINE_AA)
cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

