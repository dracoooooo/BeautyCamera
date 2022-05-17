import cv2 as cv
import numpy as np
import pyrender
import trimesh

from config import conf


class SpecialEffect:
    def __init__(self, model_path: str, landmark: int):
        """
        :param model_path: path to the 3d model
        :param landmark: which landmark the effect apply to
        """
        self.landmark = landmark

        fuze_trimesh = trimesh.load(model_path)
        try:
            mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
            scene = pyrender.Scene(ambient_light=(255, 255, 255))
            scene.add(mesh)
        except:
            scene = pyrender.Scene.from_trimesh_scene(fuze_trimesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 400.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.array((255, 255, 255)), intensity=300)
        scene.add(light, pose=camera_pose)
        self.scene = scene
        self.r = pyrender.OffscreenRenderer(800, 800)

        color, depth = self.r.render(self.scene)
        self.color = color
        self.depth = depth

        sum = np.sum(self.depth, axis=0)
        idx = np.where(sum > 0)[0]
        left, right = idx[0], idx[-1]
        self.size = right - left

    def render(self):
        """
        Render the scene
        """
        color, depth = self.r.render(self.scene)
        if conf.getboolean('Common', 'Debug'):
            cv.imshow('3d model', color)
        self.color = color
        self.depth = depth

    def scale(self, size: float):
        """
        Scale up or down according to size
        call after rotate
        :param size: float, > 0
        :return: None
        """
        for mesh_node in self.scene.mesh_nodes:
            mesh_pose = self.scene.get_pose(mesh_node)
            scaling_matrix = np.eye(4) * size
            scaling_matrix[3, 3] = 1
            self.scene.set_pose(mesh_node, np.dot(mesh_pose, scaling_matrix))
        self.render()

    def rotate(self, rvec: np.ndarray):
        rotation_matrix_3x3, _ = cv.Rodrigues(rvec)

        x_180 = np.array(
            [[1, 0, 0],
             [0, -1, 1],
             [0, 0, -1]]
        )
        rotation_matrix_3x3 = np.dot(rotation_matrix_3x3, x_180)
        # # Reverse rotation matrix to fit face direction
        rotation_matrix_3x3[1, 2] = -rotation_matrix_3x3[1, 2]
        rotation_matrix_3x3[2, 1] = -rotation_matrix_3x3[2, 1]
        rotation_matrix_3x3[0, 2] = -rotation_matrix_3x3[0, 2]
        rotation_matrix_3x3[2, 0] = -rotation_matrix_3x3[2, 0]
        rotation_matrix_3x3[0, 1] = -rotation_matrix_3x3[0, 1]
        rotation_matrix_3x3[1, 0] = -rotation_matrix_3x3[1, 0]
        rotation_matrix_4x4 = np.eye(4)
        rotation_matrix_4x4[:3, :3] = rotation_matrix_3x3

        for mesh_node in self.scene.mesh_nodes:
            # mesh_pose = self.scene.get_pose(mesh_node)
            self.scene.set_pose(mesh_node, rotation_matrix_4x4)
        self.render()

    def get_frame(self):
        if self.color is None:
            self.render()
        return self.color

    def get_mask(self):
        if self.depth is None:
            self.render()
        mask2D = np.where(self.depth > 0, 1, 0).astype('uint8')
        mask3d = np.dstack((mask2D, mask2D, mask2D))
        return mask3d
