#!/usr/bin/env python3

from carla import image_converter, sensor
from carla.client import make_carla_client, VehicleControl
from carla.tcp import TCPConnectionError
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.util import print_over_same_line

import cv2
import pygame

import csv
import os
from datetime import datetime
from multiprocessing import Queue, Pool

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


def record(queue, done):
    converters = [image_converter.to_bgra_array,
                  image_converter.depth_to_logarithmic_grayscale,
                  image_converter.labels_to_cityscapes_palette]

    while True:
        item = queue.get(True)
        if item is None:
            break

        path, name, cameras, extra = item

        for img, cam, t in cameras:
            filename = '{}_{}.png'.format(cam, name)
            img_path = os.path.join(path, filename)

            convert = converters[t]
            array = convert(img)

            cv2.imwrite(img_path, array)
            extra[cam] = filename

        done.put(extra)


def dump_record_to_csv(done, path, fieldnames):
    with open(path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        while not done.empty():
            row = done.get()
            writer.writerow(row)


class AutoDriver:
    def __init__(self, client, city_name=None):
        self.display = None
        self.clock = None
        self.controller = None

        self.client = client
        self.city_name = city_name
        self.settings = None

        self.main_view = None
        self.second_view = None
        self.third_view = None

        self.map = CarlaMap(city_name, 16.43, 50.0) if city_name is not None else None
        self.map_view = self.map.get_map(WINDOW_HEIGHT) if city_name is not None else None
        self.map_shape = self.map.map_image.shape if city_name is not None else None

        self.info = None
        self.positions = None
        self.episode_data_dir = None
        self.recording = None

    def initialize_display(self):
        pygame.init()
        pygame.display.set_caption('Autonomous driving')
       
        if self.city_name is not None:
            self.display = pygame.display.set_mode(
                    (WINDOW_WIDTH + self.map_view.shape[1], int(1.5*WINDOW_HEIGHT)),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self.display = pygame.display.set_mode(
                    (WINDOW_WIDTH, int(1.5*WINDOW_HEIGHT)), 
                    pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

        self.clock = pygame.time.Clock()
    
    def carla_settings(self):
        settings = CarlaSettings()
        settings.set(SynchronousMode=False,
                     SendNonPlayerAgentsInfo=False,
                     NumberOfVehicles=0,
                     NumberOfPedestrians=0,
                     WeatherId=0)
        settings.randomize_seeds()

        camera0 = sensor.Camera('CameraRGB')
        camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        camera0.set_position(200, 0, 140)
        camera0.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera0)

        camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
        camera1.set_image_size(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        camera1.set_position(200, 0, 140)
        camera1.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera1)

        camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
        camera2.set_image_size(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        camera2.set_position(200, 0, 140)
        camera2.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera2)

        self.settings = settings

    def control(self):
        vcontrol = VehicleControl()

        # Press Y
        if self.controller.get_button(3):
            return None
 
        # Left stick
        steer = self.controller.get_axis(0)
        if abs(steer) >= 0.1:
            vcontrol.steer = steer / 2. 
        # Right stick
        throttle = -self.controller.get_axis(4)
        if throttle > 0:
            vcontrol.throttle = abs(throttle) / 1.5
        else:
            vcontrol.brake = abs(throttle)
  
        # Press right stick
        if self.controller.get_button(10):
            vcontrol.hand_brake = True
  
        self.client.send_control(vcontrol)

        self.info['Throttle'] = vcontrol.throttle if throttle > 0 else -vcontrol.brake
        self.info['Steer'] = vcontrol.steer

    def loop(self):
        self.clock.tick(15)

        print_over_same_line('{} FPS {}'.format(self.clock.get_fps(), self.recording))

        measurements, sensor_data = self.client.read_data()
        
        self.main_view = sensor_data['CameraRGB']
        self.second_view = sensor_data['CameraDepth']
        self.third_view = sensor_data['CameraSemSeg']

        if self.city_name is not None:
            position = self.map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self.positions.append(position)

        self.info['Speed'] = measurements.player_measurements.forward_speed

    def render(self):
        if self.main_view is not None:
            array = image_converter.to_rgb_array(self.main_view)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            self.display.blit(surface, (0, 0))

        if self.second_view is not None:
            array = image_converter.depth_to_logarithmic_grayscale(self.second_view)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            self.display.blit(surface, (0, WINDOW_HEIGHT))

        if self.third_view is not None:
            array = image_converter.labels_to_cityscapes_palette(self.third_view)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            self.display.blit(surface, (WINDOW_WIDTH // 2, WINDOW_HEIGHT))

        if self.map_view is not None:
            array = self.map_view[:, :, :3]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_scale = self.map_view.shape[0] / self.map_shape[0]
            h_scale = self.map_view.shape[1] / self.map_shape[1]

            pointlist = []
            for position in self.positions:
                w_pos = int(position[0] * w_scale)
                h_pos = int(position[1] * h_scale)

                pointlist.append((w_pos, h_pos))
   
            if len(pointlist) > 1:
                pygame.draw.lines(surface, 0xff0000, False, pointlist, 2)
            pygame.draw.circle(surface, 0xff0000, pointlist[-1], 6, 0)

            self.display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()

    def episode(self):
        self.carla_settings()
        _ = self.client.load_settings(self.settings)
        self.client.start_episode(0)

        self.positions = []
        self.info = {}
        self.recording = None
        
        dirname = datetime.now().strftime('%Y%m%d%H%M')
        self.episode_data_dir = os.path.join(os.getcwd(), 'data', dirname)
        
        os.mkdir(self.episode_data_dir)
        os.mkdir(os.path.join(self.episode_data_dir, 'IMG'))
        open(os.path.join(self.episode_data_dir, 'driving_log.csv'), 'a').close()

    def start(self):
        self.initialize_display()
        self.episode()

        queue, done = Queue(), Queue()
        workers = 5
        pool = Pool(workers, record, (queue, done))
        fieldnames = ['Center', 'Depth', 'Speed', 'Steer', 'Throttle']

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            self.recording = not self.recording

                            if not self.recording:
                                csv_path = os.path.join(self.episode_data_dir, 'driving_log.csv')
                                dump_record_to_csv(done, csv_path, fieldnames)

                self.control()
                self.loop()
                self.render()

                if self.recording:
                    path = os.path.join(self.episode_data_dir, 'IMG')
                    name = datetime.now().strftime('%Y%m%d%H%M%S%f')
                    cameras = [(self.main_view, 'Center', 0), (self.second_view, 'Depth', 1)]
                    queue.put((path, name, cameras, self.info.copy()))
        finally:
            pygame.quit()

            for _ in range(2*workers):
                queue.put(None)

            pool.close()
            pool.join()

            csv_path = os.path.join(self.episode_data_dir, 'driving_log.csv')
            dump_record_to_csv(done, csv_path, fieldnames)


def main():
    host, port = 'localhost', 2000

    while True:
        try:
            with make_carla_client(host, port) as client:
                driver = AutoDriver(client, 'Town02')
                driver.start()
                break
        except TCPConnectionError:
            # logging error
            raise


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
