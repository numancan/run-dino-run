import cv2
from PIL import ImageGrab
import pyautogui
import numpy as np
import time

game_speed = 0
restart_template = cv2.imread("assets/restart.png", 0)
dino_template = cv2.imread("assets/dino.png", 0)

ACCELERATION = 0.0064
MAX_GAME_SPEED = 4
DIS_MULT = 0.0227
GAP_MULT = 0.161
DINO_X = 60


def SetSpeed(value):
    global game_speed
    game_speed = value


def crash_info(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, restart_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.8)
    return len(loc[0]) == 0


def get_game_screen():
    return cv2.cvtColor(np.array(ImageGrab.grab(bbox=(0, 80, 500, 165 + 80))), cv2.COLOR_BGR2RGB)


def calculate_gap(rect):
    x, y, w, h = cv2.boundingRect(rect)
    gap = round((151 - (y + h)) * GAP_MULT, 1)
    if gap < 2:
        gap = 0
    return gap


def FindDino(frame):
    pass


def ProcessScreen(frame):
    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = cv2.GaussianBlur(processed, (5, 5), 0)
    _, processed = cv2.threshold(processed, 150, 255, cv2.THRESH_BINARY_INV)
    return processed


def obstacle_values():
    global game_speed
    if game_speed <= MAX_GAME_SPEED:
        game_speed += ACCELERATION
        game_speed = round(game_speed, 4)
    is_dino_alive = True
    is_grounded = True
    detecting = []
    det_dis = []
    obstacle_fea = [440 * DIS_MULT, 0, game_speed]  # [0]distance, [1]gap, [2]speed
    game_screen = get_game_screen()
    processed = ProcessScreen(game_screen)

    p, conts, __ = cv2.findContours(processed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)

    for cnt in conts:
        if(cv2.contourArea(cnt) > 50):
            x, y, w, h = cv2.boundingRect(cnt)

            if(w == 40 or w == 55):  # dino
                if(y == 107) or y == 125:
                    is_grounded = True
                else:
                    is_grounded = False
#                cv2.rectangle(game_screen, (x, y), (x + w, y + h), (255, 244, 0), 2)

            else:
                detecting.append(cnt)
                det_dis.append(x - DINO_X)

    if len(detecting) > 0:
        minDis = min(det_dis)
        near = detecting[det_dis.index(minDis)]
        distance = round(minDis * DIS_MULT, 1)
        gap = calculate_gap(near)

        obstacle_fea = [distance, gap, game_speed]

    is_dino_alive = crash_info(game_screen)
    # cv2.imshow("processed", game_screen)
    return obstacle_fea, is_dino_alive, is_grounded
