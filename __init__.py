# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Quick Braids",
    "description":  "Tool for Quick Braids",
    "author": "Noizirom",
    "version": (0, 0, 1),
    "blender": (2, 93, 0),
    "location": "View3D > UI > Quick Braids",
    "warning": "", 
    "wiki_url": "",
    "tracker_url": "",
    "category": "Quick Braids"
}


import bpy
import bmesh
import numpy as np
from mathutils import geometry

# COLLECTION
####################
def collection_object_list(collection):
    return [o.name for o in bpy.data.collections[collection].objects[:]]

def new_collection(Name):
    if bpy.data.collections.get(Name) == None:
        new_coll = bpy.data.collections.new(Name)
        bpy.context.scene.collection.children.link(new_coll)
        return new_coll
    else:
        new_coll = bpy.data.collections.get(Name)
        return new_coll

def new_subcollection(Name, collection):
    if bpy.data.collections.get(Name) == None:
        new_coll = bpy.data.collections.new(Name)
        bpy.data.collections[collection].children.link(new_coll)
        return new_coll
    else:
        new_coll = bpy.data.collections.get(Name)
        return new_coll

def new_collect_object(ob, collection):
    bpy.data.collections[collection].objects.link(ob)

def if_collection_exists(Name):
    if bpy.data.collections.get(Name):
        return True
    return False

def get_collection(Name, collection=None):
    try:
        if if_collection_exists(Name):
            return bpy.data.collections.get(Name)
        coll = bpy.data.collections.new(Name)
        if collection==None:
            collection = bpy.context.collection
        collection.children.link(coll)
        return coll
    except Exception as e:
        print(e)
        return

def remove_collection(Name):
    if isinstance(Name, str):
        Name = get_collection(Name)
    try:
        bpy.data.collections.remove(Name)
    except Exception as e:
        print(e)

def show_collection(collection, mode=False):
    bpy.data.collections[collection].hide_viewport = not mode

# OBJECT
####################

def create_instance(Name, ob, collection=None):
    try:
        instance = bpy.data.objects.new(Name, ob.data)
        if collection==None:
            collection = bpy.context.collection
        collection.objects.link(instance)
        return instance
    except Exception as e:
        print(e)

def apply_instance(ob, preserve_all_data_layers=False, depsgraph=None):
    try:
        mesh = bpy.data.meshes.new_from_object(ob, preserve_all_data_layers=preserve_all_data_layers, depsgraph=depsgraph)
        mesh.name = ob.name
        ob.data = mesh
        return ob
    except Exception as e:
        print(e)

def get_eval(ob):
    deps = bpy.context.evaluated_depsgraph_get()
    return ob.evaluated_get(deps)

def get_co(ob):
    vt = ob.data.vertices
    ct = len(vt)
    co = np.empty(ct*3)
    vt.foreach_get('co', co)
    return co.reshape((ct, 3))

def get_selected_verts(ob):
    vt = ob.data.vertices
    ct = len(vt)
    co = np.empty(ct, bool)
    vt.foreach_get('select', co)
    return co

def get_face_verts(ob):
    faces = ob.data.polygons
    if len(faces) > 0:
        return np.array([i.vertices[:] for i in faces])
    return []

def get_face_centers(ob):
    faces = ob.data.polygons
    ct = len(faces)
    co = np.empty(ct*3)
    faces.foreach_get('center', co)
    return co.reshape((ct,3))

def get_face_select(ob):
    faces = ob.data.polygons
    ct = len(faces)
    co = np.empty(ct, bool)
    faces.foreach_get('select', co)
    return co

def get_face_sel_idx(sel):
    idx = np.arange(len(sel))
    return idx[sel]

def get_face_norms(ob):
    faces = ob.data.polygons
    ct = len(faces)
    co = np.empty(ct*3)
    faces.foreach_get('normal', co)
    return co.reshape((ct,3))

def get_face_verts_dict(ob):
    fa = ob.data.polygons
    ct = len(fa)
    return {i: np.array(fa[i].vertices[:]) for i in np.arange(ct)}

def get_modified_co(ob):
    import bmesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    bm.from_object(ob, depsgraph)
    bm.verts.ensure_lookup_table()
    co = array([v.co for v in bm.verts], float)
    bm.free()
    return co

def find_faces(f_verts, idxs):
    return np.array([all(np.isin(list(fv), list(idxs))) for fv in f_verts])

def selected_vert_convert(ob, vert_idx):
    vertices = ob.data.vertices
    faces = ob.data.polygons
    countv = len(vertices)
    co = get_co(ob)
    countf = len(faces)
    f_verts = get_face_verts(ob)
    mask = find_faces(f_verts, vert_idx)
    face_idx = f_verts[mask.tolist()]
    nv_Dict = {o: n for n, o in enumerate(vert_idx)}
    new_faces = [[nv_Dict[i] for i in nest] for nest in face_idx]
    return co[vert_idx], new_faces

def get_edge_vecs_dict(ob):
    ed = get_edges(ob)
    co = get_co(ob)
    shape = ed.shape
    rav = ed.ravel()
    dists = co[rav]
    vecs = np.array([(dists[i]-dists[i-1]) for i in range(1,shape[0]*2,2)])
    return {
        "vecs": vecs, 
        "dists": np.array([np.linalg.norm(i) for i in vecs]),
        }

def get_connected_verts(ob):
    vt = ob.data.vertices
    ct = len(vt)
    idx = np.arange(ct)
    edges_flat = get_edges(ob).ravel()
    arg = lambda i: np.argwhere(edges_flat == i)[:,0]//2
    return {int(i): arg(i) for i in idx}

def get_connected_face_verts(ob):
    vt = ob.data.vertices
    ct = len(vt)
    idx = np.arange(ct)
    f_verts = get_face_verts(ob)
    f_vert_ct = len(f_verts)
    mask = lambda v: np.argwhere(f_vert_ct==v)[:,0]
    arg = lambda i, v, flat: np.argwhere(flat == i)[:,0]//v
    connected = {int(i): [] for i in idx}
    for v in np.unique(f_vert_ct):
        m = mask(v).tolist()
        fv = np.array(f_verts)[m]
        f_v = np.array([i for i in fv])
        flat = f_v.ravel()
        for i in idx:
            data = arg(i, v, flat)
            connected[i].extend( data)
    return connected

def deselect_all():
    for ob in bpy.context.selected_objects:
        ob.select_set(False)

def select(ob):
    deselect_all()
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob

def remove(ob):
    if ob.type == 'MESH':
        bpy.data.meshes.remove(ob.data)
    if ob.type == 'CURVE':
        bpy.data.curves.remove(ob.data)
    return None

def active_ob(object, objects):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[object].select_set(state=True)
    bpy.context.view_layer.objects.active = bpy.data.objects[object]
    if objects is not None:
        for o in objects:
            bpy.data.objects[o].select_set(state=True)

def proximity_map(obs, ob, count=10, index=0):
    return np.sum(np.array([proximity_weights(get_co(ob), get_co(obj)) for obj in obs]), axis=0)

def hide_mod(mod, toggle=False):
    mod.show_render = toggle
    mod.show_viewport = toggle


def hide_all_mods(ob, toggle=False):
    mods = ob.modifiers
    for mod in mods:
        hide_mod(mod, toggle=toggle)

def copy_mesh_deps(ob):
    ob = get_eval(ob)
    me = ob.data
    co = get_co(ob)
    edges = me.edge_keys
    faces = get_face_verts(ob)
    mesh = new_mesh(f"[FINALIZED]_{me.name}", co, edges, faces)
    return mesh


def copy_ob_deps(ob, name, collection=None):
    if collection==None:
        collection = bpy.context.collection
    mesh = copy_mesh_deps(ob)
    mesh.name = name
    obj = bpy.data.objects.new(name, mesh)
    return obj

def finalize_mesh(ob, destroy=False):
    me = ob.data
    name = me.name
    mesh = copy_mesh_deps(ob)
    if destroy:
        bpy.data.meshes.remove(me)
    else:
        me.name = f"[ARCHIVE]_{name}"
    mesh.name = name
    ob.data = mesh
    return ob


# NEW OBJECT
####################

def new_mesh(Name, verts=None, edges=None, faces=None):
    mesh = bpy.data.meshes.new(Name)
    mesh.from_pydata(verts, edges, faces)
    return mesh

def new_object(Name, mesh_data):
    ob = bpy.data.objects.new(Name, mesh_data)
    return ob

def link_ob(ob, collection=None):
    if not collection:
        collection = bpy.context.collection
    collection.objects.link(ob)
    return ob

def create_ob(Name, verts=None, edges=None, faces=None, collection=None):
    mesh_data = new_mesh(Name, verts, edges, faces)
    ob = new_object(Name, mesh_data)
    link_ob(ob, collection=collection)
    return ob

def add_empty(Name, radius, collection=None, location=(0, 0, 0), rotation=(0, 0, 0), type='PLAIN_AXES'):
    if collection==None:
            collection = bpy.context.collection
    bpy.ops.object.empty_add(type=type, radius=radius, location=location, rotation=rotation)
    emp = bpy.context.object
    emp.name = Name
    emp.select_set(state=False)
    bpy.context.collection.objects.unlink(emp)
    collection.objects.link(emp)
    return emp

def mod_mesh(ob, verts=[], edges=[], faces=[]):
    mesh = ob.data
    mesh.clear_geometry()
    mesh.from_pydata(verts, edges, faces)
    return mesh


# ARMATURE
####################

def new_armature(Name):
    arm = bpy.data.armatures.new(Name)
    return arm

def new_bone(arm, Name, head=[0,0,0], tail=[0,-1,0]):
    eb = arm.data.edit_bones
    bone = eb.new(Name)
    bone.head = head
    bone.tail = tail
    return bone

def create_armature(Name, collection=None):
    arm = new_armature(Name)
    ob = new_object(Name, arm)
    link_ob(ob, collection=collection)
    return ob

def add_bone(arm, Name='root', head=[0,0,0], tail=[0,-1,0], parent=None, use_connect=False):
    bpy.ops.object.select_all(action='DESELECT')
    arm.select_set(1)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.editmode_toggle()
    bone = new_bone(arm, Name, head, tail)
    bone.use_connect = use_connect
    bone.parent = parent
    bpy.ops.object.editmode_toggle()
    return arm

# CURVE
####################
def get_curve_bez_points(ob, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.bezier_points
    count = len(bp)
    co = np.empty(count*3)
    bp.foreach_get('co', co)
    return co.reshape((count, 3))

def get_curve_bez_handle_left(ob, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.bezier_points
    count = len(bp)
    co = np.empty(count*3)
    bp.foreach_get('handle_left', co)
    return co.reshape((count, 3))

def get_curve_bez_handle_right(ob, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.bezier_points
    count = len(bp)
    co = np.empty(count*3)
    bp.foreach_get('handle_right', co)
    return co.reshape((count, 3))

def get_curve_bez_radius(ob, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.bezier_points
    count = len(bp)
    rad = np.empty(count)
    bp.foreach_get('radius', rad)
    return rad

def set_curve_bez_radius(ob, rad, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.bezier_points
    bp.foreach_set('radius', rad)
    ob.data.update_tag()

def get_curve_bez_tilt(ob, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.bezier_points
    count = len(bp)
    tilt = np.empty(count)
    bp.foreach_get('tilt', tilt)
    return tilt

def set_curve_bez_tilt(ob, tilt, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.bezier_points
    bp.foreach_set('tilt', tilt)
    ob.data.update_tag()

def curve_rot90(curve, spline=0):
    deg90 = 1.5707963267948966
    sp = curve.data.splines[spline]
    bp = sp.bezier_points
    count = len(bp)
    set_curve_bez_tilt(curve, [deg90 for _ in range(count)], spline=spline)

def get_curve_nurb_points(ob, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.points
    count = len(bp)
    co = np.empty(count*4)
    bp.foreach_get('co', co)
    return co.reshape((count, 4))

def get_curve_nurb_radius(ob, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.points
    count = len(bp)
    rad = np.empty(count)
    bp.foreach_get('radius', rad)
    return rad

def set_curve_nurb_radius(ob, rad, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.points
    bp.foreach_set('radius', rad)
    ob.data.update_tag()

def get_curve_nurb_tilt(ob, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.points
    count = len(bp)
    tilt = np.empty(count)
    bp.foreach_get('tilt', tilt)
    return tilt

def set_curve_nurb_tilt(ob, tilt, spline=0):
    sp = ob.data.splines[spline]
    bp = sp.points
    bp.foreach_set('tilt', tilt)
    ob.data.update_tag()

def new_curve(Name):
    curve = bpy.data.curves.new(name=Name, type='CURVE')
    curve.dimensions = '3D'
    return bpy.data.objects.new(Name, curve)

def new_curve_spline(curve, type):
    return curve.data.splines.new(type)

def add_poly_points(curve, points):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    count = points.shape[0]
    poly = new_curve_spline(curve, 'POLY')
    try:
        poly.points.add((count-1))
        cvco = np.hstack((points ,np.ones(count).reshape((count,1))))
        for point in range(count):
            poly.points[point].co = cvco[point]
    except Exception as e:
        print(e)
    return curve

def add_nurb_points(curve, points):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    count = points.shape[0]
    poly = new_curve_spline(curve, 'NURB')
    try:
        poly.points.add((count-1))
        cvco = np.hstack((points ,np.ones(count).reshape((count,1))))
        for point in range(count):
            poly.points[point].co = cvco[point]
    except Exception as e:
        print(e)
    return curve

def add_bez_points(curve, bezpoints):
    points = np.array(bezpoints["points"])
    count = points.shape[0]
    hl = bezpoints['handle_left']
    hr = bezpoints['handle_right']
    poly = new_curve_spline(curve, 'BEZIER')
    try:
        poly.bezier_points.add(int((count-1)))
        for point in range(count):
            poly.bezier_points[point].co = points[point]
            poly.bezier_points[point].handle_left = hl[point]
            poly.bezier_points[point].handle_right = hr[point]
    except Exception as e:
        print(e)
    return curve

def new_blank_curve(Name, location=[0,0,0], collection=None):
    if collection==None:
        collection = bpy.context.collection
    curveob = new_curve(Name)
    curveob.location = location
    collection.objects.link(curveob)
    return curveob

def new_blank_bez_curve(Name, location=[0,0,0], collection=None):
    if collection==None:
        collection = bpy.context.collection
    curve = new_blank_curve(Name, location, collection)
    poly = new_curve_spline(curve, 'BEZIER')
    return curve

def new_bez_curve(Name, points, location=[0,0,0], collection=None):
    if collection==None:
        collection = bpy.context.collection
    curve = new_blank_curve(Name, location, collection)
    add_bez_points(curve, points)
    return curve

def add_curve_points(curve, points):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    count = points.shape[0]
    poly = new_curve_spline(curve, 'NURBS')
    try:
        poly.points.add((count-1))
        cvco = np.hstack((points ,np.ones(count).reshape((count,1))))
        for point in range(count):
            poly.points[point].co = cvco[point]
    except Exception as e:
        print(e)
    return curve

def add_curve_strands(curve, points, target, mat, depth=0.0005):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    count = points.shape[0]
    poly = new_curve_spline(curve, 'NURBS')
    try:
        poly.points.add((count-1))
        cvco = np.hstack((points ,np.ones(count).reshape((count,1))))
        for point in range(count):
            poly.points[point].co = cvco[point]
            poly.points[0].radius = depth/2
            poly.points[-1].radius = 0
        curve.data.bevel_depth = depth
        mod_surface(curve, target)
        curve.data.materials.append(mat)
        curve.material_slots[0].material = mat
    except Exception as e:
        print(e)
    return curve

def add_curve_points_multiple(curve, points, depth=0.0005):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    ct = points.shape[0]
    count = points.shape[1]
    try:
        for p in range(ct):
            poly = new_curve_spline(curve, 'NURBS')
            poly.points.add((count-1))
            cvco = np.hstack((points ,np.ones(count).reshape((count,1))))
            for point in range(count):
                poly.points[point].co = cvco[point]
            poly.points[0].radius = depth/2
            poly.points[-1].radius = 0
        curve.data.bevel_depth = depth
    except Exception as e:
        print(e)
    return curve

def new_bez_curve(Name, points, location=[0,0,0], collection=None):
    if collection==None:
        collection = bpy.context.collection
    curve = new_blank_curve(Name, location, collection)
    add_bez_points(curve, points)
    return curve

def new_path_curve(Name, points, location=[0,0,0], collection=None):
    if collection==None:
        collection = bpy.context.collection
    curve = new_blank_curve(Name, location, collection)
    add_curve_points(curve, points)
    return curve

def new_path_curve_strands(Name, points, target, mat, location=[0,0,0], collection=None, depth=0.0005):
    if collection==None:
        collection = bpy.context.collection
    try:
        curve = new_blank_curve(Name, location, collection)
        add_curve_strands(curve, points, target, mat, depth)
    except Exception as e:
        print(e)
    return curve

def new_path_curve_multiple(Name, points, location=[0,0,0], collection=None, depth=0.0005):
    if collection==None:
        collection = bpy.context.collection
    curve = new_blank_curve(Name, location, collection)
    add_curve_points_multiple(curve, points, depth)
    return curve

def interpolate_spline(spline, count):
    points = spline.bezier_points
    ct = len(points)
    cpoints = np.array(geometry.interpolate_bezier(points[0].co, points[0].handle_right, points[1].handle_left, points[1].co, count))
    if ct <= 2:
        return cpoints
    else:
        nct = ct-1
        pcount = int(count/(nct))
        for p in range(2, ct, 1):
            points = spline.bezier_points
            pts = np.array(geometry.interpolate_bezier(points[p-1].co, points[p-1].handle_right, points[p].handle_left, points[p].co, count+1))
            cpoints = np.append(cpoints, pts[1:], axis=0)
    return cpoints

def proximity_spline_map(obs, co, count=10, thresh=0.1, outer_thresh=0.1, invert=True, index=0):
    return np.sum(np.array([proximity_weights(co, interpolate_spline(ob.data.splines[index], count), thresh, outer_thresh, invert) for ob in obs]), axis=0)

def move_bez_origin(ob, point):
    point = np.array(point, float)
    loc = np.array(ob.location, float)
    mw = np.array(ob.matrix_world, float)
    move = point - loc
    bp = ob.data.splines[0].bezier_points
    for _ in bp:
        co = array(_.co, float)
        hl = array(_.handle_left, float)
        hr = array(_.handle_right, float)
        _.co = co - move
        _.handle_left = hl - move
        _.handle_right = hr - move
    trans = mw[:3:,3]
    mw[:3:,3] = move
    ob.matrix_world = mw
    ob.location = point

def curve_change_type(curve, type='NURBS', spline=0):
    '''
    types = ['POLY', 'BEZIER', 'BSPLINE', 'CARDINAL', 'NURBS']
    '''
    if isinstance(type, int):
        types = ['POLY', 'BEZIER', 'BSPLINE', 'CARDINAL', 'NURBS']
        type = types[type]
    curve.data.splines[spline].type = type


#########################


class Braid:
    def __init__(self, scale=[1,1,1]):
        self.__data = np.array(
            [[-0.17187496172672678, 0.0, 0.004948377009091689], 
            [-0.15973039285728866, 0.370506921849038, 0.3174456723644514], 
            [0.10938154472971924, 0.49886302863044957, -0.008938187132831732], 
            [0.130574119510592, 0.954904898570723, 0.22611645689510876], 
            [-0.16799986936056077, 1.0, 0.0037567003743636434], 
            [-0.16580267729200773, 0.0, 0.22073288725408147], 
            [-0.07895480283650883, 0.08641553956129644, 0.18199176155036353], 
            [0.11842955064115389, 0.21308884706612397, 0.0053505015329721696], 
            [0.130574119510592, 0.5466566509763257, 0.2404051455609127], 
            [-0.16799986936056077, 0.7142180617189701, 0.018045389040167545], 
            [-0.16386513110892473, 1.0, 0.2182838060167627], 
            [0.12450183507587295, 0.0, 0.07809421553057984], 
            [0.130574119510592, 0.26088410240499044, 0.2546938342267166], 
            [-0.16799986936056077, 0.428444288402892, 0.032334077705971444], 
            [-0.15973039285728866, 0.6395372097865271, 0.26858039836864], 
            [0.11842955064115389, 0.9273073170333417, 0.0020890062829405304], 
            [0.12450183507587295, 1.0, 0.07437262450550931]]
            )
        self.__braid_edges = {
                "braid_1": [[0,1],[1,2],[2,3],[3,4]],
                "braid_2": [[5,6],[6,7],[7,8],[8,9],[9,10]],
                "braid_3": [[11,12],[12,13],[13,14],[14,15],[15,16]],
                "braid_extra": np.array([[0,1],[1,2],[2,3],[3,4],[4,5]]),
            }
        self.scale = scale
        self.len = [5, 6, 6]
        self.init_indices = [0, 5, 11]
    #
    @property
    def data(self):
        return self.__data
    #
    @property
    def braid_extra(self):
        return self.__braid_edges["braid_extra"]
    #
    @property
    def braid_1_edges(self):
        return self.__braid_edges["braid_1"]
    #
    @property
    def braid_2_edges(self):
        return self.__braid_edges["braid_2"]
    #
    @property
    def braid_3_edges(self):
        return self.__braid_edges["braid_3"]
    #
    @property
    def braid_1(self):
        return self.data[:5] * self.scale
    #
    @property
    def braid_2(self):
        return self.data[5:11] * self.scale
    #
    @property
    def braid_3(self):
        return self.data[11:] * self.scale


class BraidLink:
    def __init__(self, name):
        self.name = name
        self.__braid = Braid()
        self.braid = None
        self.__braid_1 = None
        self.__braid_2 = None
        self.__braid_3 = None
        self.braid_1 = None
        self.braid_2 = None
        self.braid_3 = None
        self.collection = None
        self.curve_guide = None
        self.cap = None
        self.end = None
        self.use_braid_1 = True
        self.use_braid_2 = True
        self.use_braid_3 = True
        self.braid_names = []
        self.braid_data = {}
        self.__mods = {
                "BR_Length": {
                    "type": 'ARRAY',
                    "data": {
                        "fit_type": 'FIT_CURVE',
                        "use_relative_offset": True,
                        "use_merge_vertices": True,
                        "relative_offset_displace": [0,1,0],
                        }
                    },
                "BR_Path_Resolution": {
                    "type": 'SUBSURF',
                    "data": {
                        "show_only_control_edges": True,
                        "levels": 4,
                        "render_levels": 4,
                        }
                    },
                "BR_Skin": {
                    "type": 'SKIN',
                    "data": {
                        "use_x_symmetry": True,
                        "use_smooth_shade": True,
                        }
                    },
                "BR_Skin_Resolution": {
                    "type": 'SUBSURF',
                    "data": {
                        "show_only_control_edges": True,
                        "levels": 3,
                        "render_levels": 3,
                        }
                    },
                "BR_Guide": {
                    "type": 'CURVE',
                    "data": {
                        "deform_axis": 'POS_Y',
                        }
                    },
                }
        coll = get_collection("BRAID")
        self.collection = get_collection(self.name, coll)
        self.curve_collection = get_collection("BR_CV_GUIDES", coll)
    #
    def add_mod(self, ob):
        mod = ob.modifiers
        for m in [*self.__mods]:
            md = mod.new(m, self.__mods[m]["type"])
            for d in self.__mods[m]["data"]:
                setattr(md, d, self.__mods[m]["data"][d])
    #
    def init_braid(self, scale=[1,1,1], use_braid=[True,True,True]):
        self.use_braid_1, self.use_braid_2, self.use_braid_3 = use_braid
        self.__braid.scale = scale
        # BRAID LINK 1
        if self.use_braid_1:
            bn1 = f'{self.name}_LINK_1'
            self.braid_names.append(bn1)
            self.__braid_1 = create_ob(bn1, self.__braid.braid_1, self.__braid.braid_1_edges, [], self.collection)
            self.add_mod(self.__braid_1)
        else:
            self.braid_names.append(None)
        # BRAID LINK 2
        if self.use_braid_2:
            bn2 = f'{self.name}_LINK_2'
            self.braid_names.append(bn2)
            self.__braid_2 = create_ob(bn2, self.__braid.braid_2, self.__braid.braid_extra.tolist(), [], self.collection)
            self.add_mod(self.__braid_2)
        else:
            self.braid_names.append(None)
        # BRAID LINK 3
        if self.use_braid_3:
            bn3 = f'{self.name}_LINK_3'
            self.braid_names.append(bn3)
            self.__braid_3 = create_ob(bn3, self.__braid.braid_3, self.__braid.braid_extra.tolist(), [], self.collection)
            self.add_mod(self.__braid_3)
        else:
            self.braid_names.append(None)
    #
    def __link(self, ob):
        try:
            self.collection.objects.link(ob)
        except Exception as e:
            print(e)
    #
    def __unlink(self, ob):
        try:
            self.collection.objects.unlink(ob)
        except Exception as e:
            print(e)
    #
    def set_guide_curve(self, curve):
        self.curve_guide = curve
        self.curve_collection.objects.link(self.curve_guide)
        self.braid_data = {'name': self.name, 'braids': self.braid_names, 'curve': self.curve_guide.name, 'collection': self.collection.name}
        if self.use_braid_1:
            self.__link(self.__braid_1)
            self.__braid_1.modifiers["BR_Length"].curve = self.curve_guide
            self.__braid_1.modifiers["BR_Guide"].object = self.curve_guide
            #self.__unlink(self.__braid_1)
        if self.use_braid_2:
            self.__link(self.__braid_2)
            self.__braid_2.modifiers["BR_Length"].curve = self.curve_guide
            self.__braid_2.modifiers["BR_Guide"].object = self.curve_guide
            #self.__unlink(self.__braid_2)
        if self.use_braid_3:
            self.__link(self.__braid_3)
            self.__braid_3.modifiers["BR_Length"].curve = self.curve_guide
            self.__braid_3.modifiers["BR_Guide"].object = self.curve_guide
            #self.__unlink(self.__braid_3)
    #
    def __update_braid(self, ob, verts, thickness=[.125,.125]):
        self.__link(ob)
        vt = ob.data.vertices
        sv = ob.data.skin_vertices[-1].data
        for i, v in enumerate(vt):
            v.co = verts[i]
            sv[i].radius[:] = thickness
        #self.__unlink(ob)
    #
    def update_braids(self, scale=[1,1,1]):
        self.__braid.scale = scale
        thickness=[.125*self.__braid.scale[0],.125*self.__braid.scale[2]]
        if self.use_braid_1:
            self.__update_braid(self.__braid_1, self.__braid.braid_1, thickness)
        if self.use_braid_2:
            self.__update_braid(self.__braid_2, self.__braid.braid_2, thickness)
        if self.use_braid_3:
            self.__update_braid(self.__braid_3, self.__braid.braid_3, thickness)
    #
    def destroy(self, ob):
        remove(ob)

class MeshBraid:
    def __init__(self, braid_data):
        self.__braid_data = braid_data
        self.__braid_1 = None
        self.__braid_2 = None
        self.__braid_3 = None
    #
    def create_braid(self):
        obj = bpy.data.objects
        braids = self.__braid_data['braids']
        collection = bpy.data.collections[self.__braid_data['collection']]
        name = self.__braid_data['name']
        mesh_names = []
        # BRAID LINK 
        if braids[0] != None:
            br1 = obj[braids[0]]
            collection.objects.link(br1)
            b1 = get_eval(br1)
            verts, faces = get_data(b1)
            nbr1 = f'{name}_BR_1'
            mesh_names.append(nbr1)
            if obj.get(nbr1) != None:
                remove(obj[nbr1].data)
            self.__braid_1 = create_ob(nbr1, verts, [], faces.tolist(), collection)
            collection.objects.unlink(br1)
        else:
            mesh_names.append(None)
        # BRAID LINK 
        if braids[1] != None:
            br2 = obj[braids[1]]
            collection.objects.link(br2)
            b2 = get_eval(br2)
            verts, faces = get_data(b2)
            nbr2 = f'{name}_BR_2'
            mesh_names.append(nbr2)
            if obj.get(nbr2) != None:
                remove(obj[nbr2].data)
            self.__braid_2 = create_ob(nbr2, verts, [], faces.tolist(), collection)
            collection.objects.unlink(br2)
        else:
            mesh_names.append(None)
        # BRAID LINK 
        if braids[2] != None:
            br3 = obj[braids[2]]
            collection.objects.link(br3)
            b3 = get_eval(br3)
            verts, faces = get_data(b3)
            nbr3 = f'{name}_BR_3'
            mesh_names.append(nbr3)
            if obj.get(nbr3) != None:
                remove(obj[nbr3].data)
            self.__braid_3 = create_ob(nbr3, verts, [], faces.tolist(), collection)
            collection.objects.unlink(br3)
        else:
            mesh_names.append(None)
        #
        self.__braid_data.update({'mesh': mesh_names})
    #
    def set_materials(self, materials=[None,None,None]):
        if materials[0] != None:
            self.__braid_1.active_material = materials[0]
        if materials[1] != None:
            self.__braid_2.active_material = materials[1]
        if materials[2] != None:
            self.__braid_3.active_material = materials[2]
    #
    def destroy(self, ob):
        remove(ob)

def spline_dict(curve):
    from numpy import array, empty
    d = {}
    for i, sp in enumerate(curve.data.splines):
        s = sp.bezier_points
        ct = len(s)
        co = empty(ct*3, float)
        hl = empty(ct*3, float)
        hr = empty(ct*3, float)
        s.foreach_get('co', co)
        s.foreach_get('handle_left', hl)
        s.foreach_get('handle_right', hr)
        d.update({i: {'points': co.reshape((ct, 3)), 'handle_left': hl.reshape((ct, 3)), 'handle_right': hr.reshape((ct, 3)), 'count': ct}})
    return d

def curve_split(curve):
    curve_dict = spline_dict(curve)
    for spline in curve_dict:
        curve = new_bez_curve(f'CG_{curve.name}_{spline}', curve_dict[spline], curve.location)
        blk = BraidLink(f'BR_{curve.name}_{spline}')
        blk.init_braid()
        blk.update_braids()
        blk.set_guide_curve(curve)
        bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = 'SURFACE'
        blk.create_braid()

class BraidSeries:
    def __init__(self, Name, scale=[1,1,1], use_braid=[True,True,True]):
        self.name = Name
        self.scale = scale
        self.use_braid = use_braid
        self.curve_dict = {}
        self.braidlink_dict = {}
        self.curve = None
        self.coll = get_collection("BRAID")
        self.collection = get_collection("CURVE_GUIDES")
        self.init_curve()
    #
    def init_curve(self):
        self.curve = new_blank_curve(self.name, )
        bpy.context.collection.objects.unlink(self.curve)
        bpy.data.collections["CURVE_GUIDES"].objects.link(self.curve)
        select(self.curve)
        bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = 'SURFACE'
        bpy.ops.object.editmode_toggle()
        bpy.ops.wm.tool_set_by_id(name="builtin.draw")
    #
    def draw_curve_guide(self):
        pass
    #
    def curve_split(self):
        if self.curve.data.is_editmode:
            bpy.ops.object.editmode_toggle()
        self.curve_dict = spline_dict(self.curve)
        for spline in self.curve_dict:
            c_name = f'CG_{self.curve.name}_{spline}'
            b_name = f'BR_{self.curve.name}_{spline}'
            curve = new_bez_curve(c_name, self.curve_dict[spline], self.curve.location)
            braidlink = BraidLink(b_name)
            braidlink.init_braid(self.scale, self.use_braid)
            braidlink.update_braids(self.scale)
            braidlink.set_guide_curve(curve)
            self.braidlink_dict.update({b_name: braidlink.braid_data})
            bpy.context.collection.objects.unlink(curve)
        ob = self.curve
        self.curve = None
        remove(ob)
    #
    def braids_to_mesh(self):
        for b in self.braidlink_dict:
            mb = MeshBraid(b)
            mb.create_braid()
    

braid_ob = None
ct = 0

def braid_draw():
    global braid_ob
    global ct
    braid_ob = BraidSeries(f"Braid_{ct}")
    ct += 1


class DrawBraid(bpy.types.Operator):
    """Draw curve guide for braids"""
    bl_idname = "curve.draw_braid"
    bl_label = "Draw Guide"

    def execute(self, context):
        braid_draw()
        return {'FINISHED'}

class MakeBraid(bpy.types.Operator):
    """create braids along guides"""
    bl_idname = "curve.make_braid"
    bl_label = "Make Braid"

    @classmethod
    def poll(cls, context):
       return (context.active_object is not None)

    def execute(self, context):
        braid_ob.curve_split()
        return {'FINISHED'}

class BraidPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Quick_Braids"
    bl_idname = "OBJECT_PT_braid"
    bl_space_type = "VIEW_3D"
    bl_region_type = 'UI'
    bl_category = "Quick Braids"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("curve.draw_braid")
        row.operator("curve.make_braid")

classes = [
    DrawBraid,
    MakeBraid,
    BraidPanel,
]

rev_classes = classes.copy()
rev_classes.reverse()

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in rev_classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()



