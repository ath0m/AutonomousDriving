from carla import image_converter
import cv2
import csv
import os


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


def read_simulator_data(client, target):
    while target.connect:
        target.data = client.read_data()
