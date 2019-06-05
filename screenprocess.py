import cv2
from PIL import ImageGrab
import pyautogui
import numpy as np
import time


replay_but_template = cv2.imread("assets/restart.png", 0)
dino_template = cv2.imread("assets/dino.png", 0)

game_speed = 0

GRAB_WIDTH = 500
GRAB_HIGHT = 165

ACCELERATION = 0.0064
MAX_GAME_SPEED = 5.4
# For scaling
DIS_MULT = 10 / GRAB_WIDTH
GAP_MULT = 10 / GRAB_HIGHT


class Screen():
    def rgb(self):
        return cv2.cvtColor(np.array(ImageGrab.grab(bbox=(0, 80, GRAB_WIDTH, GRAB_HIGHT + 80))), cv2.COLOR_BGR2RGB)

    def gray(self):
        return cv2.cvtColor(self.rgb(), cv2.COLOR_BGR2GRAY)


def is_dino_alive(grayframe):
    res = cv2.matchTemplate(grayframe, replay_but_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.8)
    return len(loc[0]) == 0


def update_speed():
    global game_speed
    if game_speed <= MAX_GAME_SPEED:
        game_speed += ACCELERATION
        game_speed = round(game_speed, 2)


def reset_speed():
    global game_speed
    game_speed = 0


def process_screen(frame):
    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = cv2.GaussianBlur(processed, (5, 5), 0)
    _, processed = cv2.threshold(processed, 150, 255, cv2.THRESH_BINARY_INV)
    return processed


def get_near_obs(obstacles):
    all_x = [cv2.boundingRect(cnt)[0] for cnt in obstacles]
    return obstacles[all_x.index(min(all_x))]


def get_obstacles(processed):
    # Find all contours in processed frame
    try:
        p, conts, __ = cv2.findContours(processed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
    except ValueError:
        conts, __ = cv2.findContours(processed, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)

    obstacles = []
    dino = [-1] * 4
    for cnt in conts:
        if(cv2.contourArea(cnt) > 100):
            x, y, w, h = cv2.boundingRect(cnt)

            if (w == 40 or w == 55):  # Dino shape
                dino = [x, y, w, h]

            # if obstacle  is on the right
            elif x > dino[0]:
                obstacles.append(cnt)

    return obstacles, dino


def make_synchronise():
    screen = Screen()
    i = 0
    while True:
        frame = screen.rgb()
        processed = process_screen(frame)
        _, dino = get_obstacles(processed)
        if dino[0] == 0 and dino[1] == 105:
            print("Synchronising!")
            i += 1
            if i == 20:
                print("Synchronised!")
                return True
                break
        else:
            print("Can't synchronise!")
            print("Dino x,y {0},{1} must be 0,105".format(
                dino[0], dino[1]))
    return False


# TODO: Night Mode
def get_inputs():

    update_speed()

    screen = Screen()
    distance = 10
    gap = 0

    frame = screen.rgb()
    processed = process_screen(frame)

    obstacles, dino = get_obstacles(processed)

    # if there is obstacle
    if obstacles != []:
        x, y, w, h = cv2.boundingRect(get_near_obs(obstacles))
        distance = round((x - dino[0]) * DIS_MULT, 1)

        gap = round((GRAB_HIGHT - (y + h)) * GAP_MULT, 1)
        gap = 0 if gap < 2 else gap

    # cv2.imshow("frame", frame)
    # cv2.imshow("processed", processed)
    # cv2.waitKey(1)

    return [distance, gap, game_speed], is_dino_alive(screen.gray())
