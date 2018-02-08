#!/usr/bin/env python3

from carla.client import make_carla_client
from carla.tcp import TCPConnectionError
from autonomous import AutoDriver


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
