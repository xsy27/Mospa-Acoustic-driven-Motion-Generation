import bpy
import math


class Camera:
    def __init__(self, first_root, mode):
        camera = bpy.data.objects['Camera']

        # initial position
        scale = 1
        camera.location.x = scale * 3.51
        camera.location.y = scale * -14.703
        camera.location.z = scale * 5.8542

        camera.rotation_euler.x = 68.626 * (math.pi / 180.0)
        camera.rotation_euler.y = 0
        camera.rotation_euler.z = 13.359 * (math.pi / 180.0)

        # wider point of view
        if mode == "sequence":
            camera.data.lens = 70
        if mode == "sep_sequence":
            camera.data.lens = 100
        elif mode == "frame":
            camera.data.lens = 130
        elif mode == "video":
            camera.data.lens = 80

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]
        self.camera.location.z += first_root[2]

        self._root = first_root

    def update(self, new_root):
        delta_root = new_root - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]

        self._root = new_root
