from screenprocess import get_inputs, make_synchronise, reset_speed
import pyautogui
import time
import cv2
import neat
import sys

pyautogui.PAUSE = 0
synchronised = False


def restart():
    pyautogui.hotkey("ctrl", "r")
    time.sleep(0.5)
    pyautogui.keyDown("num8")
    time.sleep(0.5)
    pyautogui.keyUp("num8")


def play(genome, config):
    global synchronised
    while synchronised == False:
        synchronised = make_synchronise()

    is_dino_alive = True

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    restart()

    start_time=time.time()

    # Play until the dinosaur dies
    while is_dino_alive:
        try:
            inputs, is_dino_alive = get_inputs()
            outputs = net.activate(inputs)

            # print("inputs", inputs)
            # print("outputs", outputs)

            if outputs[0] > 0.7:
                print("Duck!")
                pyautogui.keyDown("num2")
            else:
                pyautogui.keyUp("num2")

            if outputs[1] > 0.7:
                print("Low Jump!")
                lowJump()

            cv2.waitKey(1)
        except KeyboardInterrupt:
            print("Game Stopped!")
            cv2.destroyAllWindows()
            sys.exit()
    fitness = round(time.time()-start_time,2)

    reset_speed()
    cv2.destroyAllWindows()
    return fitness


def lowJump():
    pyautogui.keyDown("num8")
    time.sleep(0.1)
    pyautogui.keyUp("num8")
