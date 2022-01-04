import pyrender
import trimesh
import numpy as np
import cv2

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


        try:
            color, _ = self.r.render(self.scene)
        except:
            print('Rendering failed')
            color = np.zeros((self.height, self.width, 3))

        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        return color[..., ::-1].copy()


def render_mesh(mesh, height, width):


    rgb_per_v = np.zeros_like(mesh.v)
    rgb_per_v[:, 0] = 0.53
    rgb_per_v[:, 1] = 0.81
    rgb_per_v[:, 2] = 0.98

    tri_mesh = trimesh.Trimesh(vertices=0.001*mesh.v, faces=mesh.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
    scene.add(render_mesh, pose=np.eye(4))

    camera = pyrender.camera.OrthographicCamera(xmag=0.001*0.5*width, ymag=0.001*0.5*height, znear=0.01, zfar=10)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0.001*0.5*width, 0.001*0.5*height, 1.0])
    scene.add(camera, pose=camera_pose)

    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([1.0, 1.0, 1.0])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = np.array([0.0, 1.0, 1.0])
    scene.add(light, pose=light_pose.copy())

    try:
        r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        color, _ = r.render(scene)
    except:
        print('Rendering failed')
        color = np.zeros((height, width, 3))

    return color[..., ::-1].copy()