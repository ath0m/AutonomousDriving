#!/usr/bin/env python3

import pygame
import logging
import random
import time
import argparse
import csv

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_SPACE
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_q
from pygame.locals import K_r
from pygame.locals import K_s
from pygame.locals import K_w

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

def make_carla_settings():
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=False,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        WeatherId=-1)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(200, 0, 140)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    return settings

class CarlaSimulator:
    def __init__(self, carla_client, driving_logger=None):
        self.client = carla_client

        self._display = None
        self._logger = driving_logger
        self._clock = None
        self._main_image = None
        self._is_on_reverse = False
        self._recording = False
        self._joystick = None
        self._counter = 0

    def run(self):
        pygame.init()
        self.init()
        pygame.joystick.init()

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self.loop()
                self.render()
        finally:
            pygame.quit()

    def init(self):
        self._display = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self._clock = pygame.time.Clock()

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        logging.debug('pygame started')
        self.new_episode()

    def render(self):
        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        pygame.display.flip()
        self._clock.tick(60)

    def loop(self):
        measurements, sensor_data = self.client.read_data()

        self._main_image = sensor_data['CameraRGB']

        self.print_player_measurements(measurements.player_measurements)

        # control = self.get_keyboard_control(pygame.key.get_pressed())
        control = self.get_pad_control(self._joystick)

        if control is None:
            self.new_episode()
        else:
            self.client.send_control(control)

        if self._recording:
            self.record(self._main_image, control, measurements.player_measurements)

    def record(self, image, control, measurements):
        image_filename = 'IMG/image_{}.png'.format(self._counter)
        self._counter += 1

        image.save_to_disk(image_filename)

        steer = control.steer
        speed = measurements.forward_speed

        throttle = control.throttle
        if control.brake > throttle:
            throttle = control.brake

        self._logger.writerow([image_filename, steer, throttle, speed])

    def new_episode(self):
        scene = self.client.load_settings(make_carla_settings())
        self.client.start_episode(0)
        self._is_on_reverse = False
        self._recording = False

    def print_player_measurements(self, player_measurements):
        message = '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
        message += 'recording: {recording}'
        message = message.format(
            speed=player_measurements.forward_speed,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad,
            recording=self._recording)
        print_over_same_line(message)

    def get_keyboard_control(self, keys):
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        control.reverse = self._is_on_reverse
        return control

    def get_pad_control(self, joystick):
        control = VehicleControl()

        # Press Y
        if joystick.get_button(3):
            return None

        # Left stick
        steer = joystick.get_axis(0)
        if abs(steer) >= 0.1:
            control.steer = steer / 2.

        # Right stick
        throttle = -joystick.get_axis(4)
        if throttle > 0:
            control.throttle = abs(throttle) / 1.5
        else:
            control.brake = abs(throttle)

        # Press right stick
        if joystick.get_button(10):
            control.hand_brake = True

        # Press X
        if joystick.get_button(2):
            self._is_on_reverse = not self._is_on_reverse
        control.reverse = self._is_on_reverse

        # Press B
        if joystick.get_button(1):
            self._recording = not self._recording

        return control

def main():
    log_level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    host, port = 'localhost', 2000

    logging.info('listening to server %s:%s', host, port)

    csvfile = open('driving_log.csv', 'a')
    driving_logger = csv.writer(csvfile)

    while True:
        try:

            with make_carla_client(host, port) as client:
                simulator = CarlaSimulator(client, driving_logger)
                simulator.run()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
        finally:
            csvfile.close()


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
