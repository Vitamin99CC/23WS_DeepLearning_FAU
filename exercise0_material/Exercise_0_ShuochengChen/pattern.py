import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution: int, tile_size: int):
        if resolution % (2 * tile_size) != 0:
            raise Exception("Resolution not evenly divisible. The checker will be truncated!")
        self.output = None
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        white = np.ones((self.tile_size, self.tile_size))
        black = np.zeros((self.tile_size, self.tile_size))
        bw = np.hstack((black, white))
        wb = np.hstack((white, black))
        zebra_bw = np.tile(bw, int(self.resolution/(2 * self.tile_size)))
        zebra_wb = np.tile(wb, int(self.resolution/(2 * self.tile_size)))
        zebra_h = np.vstack((zebra_bw, zebra_wb))
        self.output = np.tile(zebra_h, (int(self.resolution/(2 * self.tile_size)), 1))

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap="gray")


class Circle:
    def __init__(self, resolution: int, radius: int, position: tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        edge = np.arange(0, self.resolution)
        xx, yy = np.meshgrid(edge, edge)
        dist = np.sqrt((xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2)
        #self.output = np.zeros((self.resolution, self.resolution))
        self.output = [dist <= self.radius]
        #for index in np.argwhere(dist <= self.radius):
        #    self.output[tuple(index)] = 1
        #self.output = np.flip(self.output, axis=0)
        #self.output = self.output.astype(bool)
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap="gray")



class Spectrum:

    def __init__(self, resolution: int):
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution, 3))

        self.output[:, :,0] = np.tile(np.linspace(0, 1, self.resolution), (self.resolution, 1))
        #for i in range(0, self.resolution):
        #    self.output[i, :, 0] = np.linspace(0, 1, self.resolution)

        #for i in range(0, self.resolution):
        #    self.output[i, :, 1] = i/(self.resolution-1)
        
        self.output[:, :, 1] = np.tile(np.linspace(0, 1, self.resolution).reshape((self.resolution, 1)), (1, self.resolution))

#       for i in range(0, 1024):
#           output[i, 0:i, 2] = np.linspace(0, 1, i)
#           output[i, i:1024, 2] = np.linspace(1, 0, 1024-i)
        self.output[:, :,2] = np.tile(np.linspace(1, 0, self.resolution), (self.resolution, 1))
        #for i in range(0, self.resolution):
        #    self.output[i, :, 2] = np.linspace(1, 0, self.resolution)
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)


