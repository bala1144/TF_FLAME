import os
import numpy as np
from numpy.core.numeric import indices
import pyrender
from pyrender import primitive
from pyrender import material
import trimesh
import cv2
from scipy.spatial.transform import Rotation
from pyrender import RenderFlags
import os

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

class flame_render:

    def __init__(self, width, height):
        self.scene = pyrender.Scene(ambient_light=[.3, .3, .3], bg_color=[255, 255, 255])
        camera = pyrender.camera.OrthographicCamera(xmag=0.001 * 0.5 * width, ymag=0.001 * 0.5 * height, znear=0.01,
                                                    zfar=10)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0.001 * 0.5 * width, 0.001 * 0.5 * height, 1.0])
        self.scene.add(camera, pose=camera_pose)

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([1.0, 1.0, 1.0])
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = np.array([0.0, 1.0, 1.0])
        self.scene.add(light, pose=light_pose.copy())

        self.r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

        self.mesh_node = None


    def render_mesh(self, vertices, faces):

        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)

        rgb_per_v = np.zeros_like(vertices)
        rgb_per_v[:, 0] = 0.53
        rgb_per_v[:, 1] = 0.81
        rgb_per_v[:, 2] = 0.98

        tri_mesh = trimesh.Trimesh(vertices=0.001 * vertices, faces=faces, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
        self.mesh_node = self.scene.add(render_mesh, pose=np.eye(4))


        color, _ = self.r.render(self.scene)

        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        return color[..., ::-1].copy()

class Facerender:
    def __init__(self, intrinsic=(2035.18464, -2070.36928, 257.55392, 256.546816),
                 img_size=(512, 512)):
        self.image_size = img_size
        self.scene = pyrender.Scene(ambient_light=[.75, .75, .75], bg_color=[0, 0, 0])

        # create camera and light
        self.add_camera(intrinsic)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        self.scene.add(light, pose=np.eye(4))
        self.r = pyrender.OffscreenRenderer(*self.image_size)
        self.mesh_node = None

    def add_camera(self, intrinsic):
        (fx, fy, Cx, Cy) = intrinsic
        camera = pyrender.camera.IntrinsicsCamera(fx, fy, Cx, Cy,
                                                  znear=0.05, zfar=10.0, name=None)

        camera_rotation = np.eye(4)
        # camera_rotation[:3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix() @ Rotation.from_euler('y', 0, degrees=True).as_matrix() @ Rotation.from_euler('x', 0, degrees=True).as_matrix()

        camera_rotation[:3, :3] = Rotation.from_euler('z', 180, degrees=True).as_dcm() @ Rotation.from_euler('y', 0, degrees=True).as_dcm() @ Rotation.from_euler( 'x', 0, degrees=True).as_dcm()

        # as_dcm
        camera_translation = np.eye(4)
        camera_translation[:3, 3] = np.array([0, 0, 1])
        camera_pose = camera_rotation @ camera_translation
        self.scene.add(camera, pose=camera_pose)

    def add_face(self, vertices, faces, pose=np.eye(4)):
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        # # disable processing
        # with Stopwatch('Creating trimesh') as f:
        #     tri_mesh = trimesh.Trimesh(vertices, faces, process=False)
        # # print(tri_mesh)
        # with Stopwatch('Creating pyrender mesh') as f:
        #     mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        # with Stopwatch('Creating mesh add') as f:
        #     self.mesh_node = self.scene.add(mesh, pose=pose)
        
                # disable processing
        tri_mesh = trimesh.Trimesh(vertices, faces)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self.mesh_node = self.scene.add(mesh, pose=pose)
    

    def add_face_v2(self, vertices, faces, pose=np.eye(4)):
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        primitive = [pyrender.Primitive(
                    positions=vertices.copy(),
                    indices=faces,
                    material = pyrender.MetallicRoughnessMaterial(
                    alphaMode='BLEND',
                    baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                    metallicFactor=0.2,
                    roughnessFactor=0.8),
                mode=pyrender.GLTF.TRIANGLES)
                    ]
        mesh = pyrender.Mesh(primitives=primitive, is_visible=True)
        self.mesh_node = self.scene.add(mesh, pose=pose)

    def add_obj(self, obj_path):

        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        mesh = trimesh.load(obj_path)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        self.mesh_node = self.scene.add(mesh)

    def render(self):
        flags = RenderFlags.SKIP_CULL_FACES
        color, _ = self.r.render(self.scene, flags=flags)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return color

# def render_mesh(mesh, height, width):
    # rgb_per_v = np.zeros_like(mesh.v)
    # rgb_per_v[:, 0] = 0.53
    # rgb_per_v[:, 1] = 0.81
    # rgb_per_v[:, 2] = 0.98

    # tri_mesh = trimesh.Trimesh(vertices=0.001*mesh.v, faces=mesh.f, vertex_colors=rgb_per_v)
    # render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
    # scene.add(render_mesh, pose=np.eye(4))

    # camera = pyrender.camera.OrthographicCamera(xmag=0.001*0.5*width, ymag=0.001*0.5*height, znear=0.01, zfar=10)
    # camera_pose = np.eye(4)
    # camera_pose[:3, 3] = np.array([0.001*0.5*width, 0.001*0.5*height, 1.0])
    # scene.add(camera, pose=camera_pose)

    # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    # light_pose = np.eye(4)
    # light_pose[:3, 3] = np.array([1.0, 1.0, 1.0])
    # scene.add(light, pose=light_pose.copy())

    # light_pose[:3, 3] = np.array([0.0, 1.0, 1.0])
    # scene.add(light, pose=light_pose.copy())

    # try:
    #     r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    #     color, _ = r.render(scene)
    # except:
    #     print('Rendering failed')
    #     color = np.zeros((height, width, 3))

    # return color[..., ::-1].copy()