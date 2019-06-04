from GetInputs import obstacle_values, SetSpeed
import pyautogui
import time
import cv2
import neat
import sys
pyautogui.PAUSE = 0
# print(neat.__file__)


def Restart():
    pyautogui.hotkey("ctrl", "r")
    time.sleep(0.5)
    pyautogui.keyDown("num8")
    time.sleep(0.5)
    pyautogui.keyUp("num8")


def Play(genome, config):
    is_dino_alive = True
    is_grounded = True
    fitness = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    Restart()
    # Play until the dinosaur dies
    while is_dino_alive:
        try:
            inputs, is_dino_alive, is_grounded = obstacle_values()
            outputs = net.activate(inputs)
            if is_grounded:
                # print("inputs", inputs)
                # print("outputs", outputs)

                if outputs[0] > 0.5:
                    print("Duck!")
                    pyautogui.keyDown("num2")
                else:
                    pyautogui.keyUp("num2")

                if outputs[1] > 0.5:
                    print("Low Jump!")
                    LowJump()

            # elif(outputs[2] > 0.4):
            #     print("High Jump!")
            #     pyautogui.press("num8")
            #     time.sleep(0.1)

            fitness += 1

            cv2.waitKey(1)
        except KeyboardInterrupt:
            print("Game Stopped!")
            cv2.destroyAllWindows()
            sys.exit()
    SetSpeed(0)
    cv2.destroyAllWindows()
    return fitness


def LowJump():
    pyautogui.keyDown("num8")
    time.sleep(0.1)
    pyautogui.keyUp("num8")
