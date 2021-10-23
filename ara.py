import numpy
import cv2

img = cv2.imread("resimler/squid.jpg")
imgri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

aranansemsi = cv2.imread("resimler/semsiye.jpg")
aranangrisemsi = cv2.cvtColor(aranansemsi, cv2.COLOR_BGR2GRAY)

aranancember = cv2.imread("resimler/cember.jpg")
aranangricember = cv2.cvtColor(aranancember, cv2.COLOR_BGR2GRAY)

arananucgen = cv2.imread("resimler/ucgen.jpg")
aranangriucgen = cv2.cvtColor(arananucgen, cv2.COLOR_BGR2GRAY)

arananyildiz = cv2.imread("resimler/yildiz.jpg")
aranangriyildiz = cv2.cvtColor(arananyildiz, cv2.COLOR_BGR2GRAY)

def semsi():
    w, h = aranangrisemsi.shape[::-1]
    res = cv2.matchTemplate(imgri, aranangrisemsi, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = numpy.where(res > threshold)

    for n in zip(*loc[::-1]):
        cv2.rectangle(img, n, (n[0] + w, n[1] + h), (255, 50, 50), 1)
        cv2.putText(
            img=img,
            text="SEMSIYE",
            org=(n[0] + w - 170, 390),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=2.0,
            color=(255, 50, 50),
            thickness=2
        )
def cember():
    w, h = aranangricember.shape[::-1]
    res = cv2.matchTemplate(imgri, aranangricember, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = numpy.where(res > threshold)

    for n in zip(*loc[::-1]):
        cv2.rectangle(img, n, (n[0] + w, n[1] + h), (0, 150, 0), 1)
        cv2.putText(
            img=img,
            text="CEMBER",
            org=(n[0] + w - 170, 390),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=2.0,
            color=(0, 150, 0),
            thickness=2
        )
def yildiz():
    w, h = aranangriyildiz.shape[::-1]
    res = cv2.matchTemplate(imgri, aranangriyildiz, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = numpy.where(res > threshold)

    for n in zip(*loc[::-1]):
        cv2.rectangle(img, n, (n[0] + w, n[1] + h), (0, 200, 255), 1)
        cv2.putText(
            img=img,
            text="YILDIZ",
            org=(n[0] + w - 155, 390),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=2.0,
            color=(0, 200, 255),
            thickness=2
        )
def ucgen():
    w, h = aranangriucgen.shape[::-1]
    res = cv2.matchTemplate(imgri, aranangriucgen, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = numpy.where(res > threshold)

    for n in zip(*loc[::-1]):
        cv2.rectangle(img, n, (n[0]+w, n[1]+h), (0, 0,255), 1)
        cv2.putText(
             img=img,
             text="UCGEN",
             org=(n[0]+w-155, 390),
             fontFace=cv2.FONT_HERSHEY_DUPLEX,
             fontScale=2.0,
             color=(0, 0, 255),
             thickness=2
        )

semsi()
cv2.imshow("Squid",img)
cv2.waitKey(3000)
cember()
cv2.imshow("Squid",img)
cv2.waitKey(3000)
ucgen()
cv2.imshow("Squid",img)
cv2.waitKey(3000)
yildiz()
cv2.imshow("Squid",img)
cv2.waitKey(3000)