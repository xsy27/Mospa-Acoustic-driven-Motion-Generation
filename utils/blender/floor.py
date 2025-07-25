import bpy
from .materials import floor_mat


def plot_floor(lims, big_plane=True):
    # Create a floor
    minx, miny, minz = lims[0]
    maxx, maxy, maxz = lims[1]

    location = ((maxx + minx)/2, (maxy + miny)/2, minz)
    # a little bit bigger
    scale = (2.25*(maxx - minx)/2, 2.25*(maxy - miny)/2, 1)

    if big_plane:
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
    else:
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                use_proportional_projected=False, release_confirm=True)
        obj = bpy.data.objects["Plane"]
        obj.name = "SmallPlane"
        obj.data.name = "SmallPlane"
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))


def show_trajectory(coords):
    for i, coord in enumerate(coords):
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Greens')
        begin = 0.45
        end = 1.0
        frac = i / len(coords)
        rgb_color = cmap(begin + (end - begin) * frac)

        x, y, z = coord
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=(x, y, z))
        obj = bpy.context.active_object

        mat = bpy.data.materials.new(name="SphereMaterial")
        obj.data.materials.append(mat)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = rgb_color

    bpy.ops.object.mode_set(mode='OBJECT')
