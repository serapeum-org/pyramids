"""Pure-NumPy/Python port of the H3 geo<->cell core (no `h3` dependency).

A faithful port of the parts of Uber's H3 (Apache-2.0) geometry engine needed by
:meth:`FeatureCollection.to_h3` and :meth:`FeatureCollection.h3_bin`:

* :func:`latlng_to_cell` — point (lat/lng degrees) -> H3 cell index (lowercase hex string).
* :func:`cell_to_boundary` — H3 cell index -> list of ``(lat, lng)`` hexagon/pentagon vertices in degrees.

Ported from the H3 4.x C source (``coordijk.h``, ``faceijk.c``, ``baseCells.c``, ``h3Index.c``, ``vec3d.h``,
``vec2d.c``, ``latLng.c``). The large constant tables live in :mod:`pyramids.feature._h3_tables`. Verified
bit-for-bit against the upstream ``h3`` library via committed fixtures (``tests/data/h3/h3_vectors.json``).
"""

from __future__ import annotations

import math

from pyramids.feature._h3_tables import (
    ADJACENT_FACE_DIR,
    BASE_CELL_DATA,
    FACE_AXES_AZ_RADS_CII,
    FACE_CENTER_POINT,
    FACE_IJK_BASE_CELLS,
    FACE_NEIGHBORS,
    MAX_DIM_BY_CII_RES,
    UNIT_SCALE_BY_CII_RES,
)

M_PI = math.pi
M_2PI = 2.0 * math.pi
M_PI_180 = math.pi / 180.0
M_180_PI = 180.0 / math.pi
EPSILON = 0.0000000000000001
M_SQRT3_2 = 0.8660254037844386467637231707529361834714
M_RSIN60 = 1.1547005383792515290182975610039149112953
M_AP7_ROT_RADS = 0.333473172251832115336090755351601070065900389
RES0_U_GNOMONIC = 0.38196601125010500003
INV_RES0_U_GNOMONIC = 2.61803398874989588842
M_SQRT7 = 2.6457513110645905905016157536392604257102
M_RSQRT7 = 0.37796447300922722721451653623418006081576
M_ONESEVENTH = 1.0 / 7.0
M_ONETHIRD = 1.0 / 3.0
FLT_EPSILON = 1.1920928955078125e-07

MAX_H3_RES = 15
NUM_BASE_CELLS = 122
MAX_FACE_COORD = 2
NUM_HEX_VERTS = 6
NUM_PENT_VERTS = 5
H3_INIT = 35184372088831
H3_CELL_MODE = 1

CENTER_DIGIT, K_AXES_DIGIT, J_AXES_DIGIT, JK_AXES_DIGIT = 0, 1, 2, 3
I_AXES_DIGIT, IK_AXES_DIGIT, IJ_AXES_DIGIT, INVALID_DIGIT = 4, 5, 6, 7
NUM_DIGITS = 7
IJ, KI, JK = 1, 2, 3
NO_OVERAGE, FACE_EDGE, NEW_FACE = 0, 1, 2

UNIT_VECS = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0))


def _lround(x: float) -> int:
    """C ``lround`` — round half away from zero (Python ``round`` is half-to-even)."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def is_class_iii(res: int) -> bool:
    return res % 2 == 1


# --- CoordIJK math (operates on mutable [i, j, k] lists, mirroring the in-place C) ---


def _ijk_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _ijk_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _ijk_scale(c, f):
    return [c[0] * f, c[1] * f, c[2] * f]


def _ijk_normalize(c):
    if c[0] < 0:
        c[1] -= c[0]
        c[2] -= c[0]
        c[0] = 0
    if c[1] < 0:
        c[0] -= c[1]
        c[2] -= c[1]
        c[1] = 0
    if c[2] < 0:
        c[0] -= c[2]
        c[1] -= c[2]
        c[2] = 0
    m = min(c)
    if m > 0:
        c[0] -= m
        c[1] -= m
        c[2] -= m
    return c


def _ijk_to_hex2d(h):
    i = h[0] - h[2]
    j = h[1] - h[2]
    return [i - 0.5 * j, j * M_SQRT3_2]


def _hex2d_round_low(r1, r2, m1, m2):
    """Quantize for the inner third of the cell (``r1 < 0.5``)."""
    if r1 < 1.0 / 3.0:
        return (m1, m2) if r2 < (1.0 + r1) / 2.0 else (m1, m2 + 1)
    j = m2 if r2 < (1.0 - r1) else m2 + 1
    i = m1 + 1 if (1.0 - r1) <= r2 < (2.0 * r1) else m1
    return i, j


def _hex2d_round_high(r1, r2, m1, m2):
    """Quantize for the outer third of the cell (``r1 >= 0.5``)."""
    if r1 < 2.0 / 3.0:
        j = m2 if r2 < (1.0 - r1) else m2 + 1
        i = m1 if (2.0 * r1 - 1.0) < r2 < (1.0 - r1) else m1 + 1
        return i, j
    return (m1 + 1, m2) if r2 < (r1 / 2.0) else (m1 + 1, m2 + 1)


def _hex2d_round(r1, r2, m1, m2):
    """Quantize sub-cell offsets ``(r1, r2)`` to integer ``(i, j)`` (DGGRID rounding)."""
    if r1 < 0.5:
        return _hex2d_round_low(r1, r2, m1, m2)
    return _hex2d_round_high(r1, r2, m1, m2)


def _hex2d_fold(h, x, y):
    """Fold ijk across the i/j axes for negative-quadrant hex2d coordinates (in place)."""
    if x < 0.0:
        if (h[1] % 2) == 0:
            diff = h[0] - h[1] // 2
            h[0] = int(h[0] - 2.0 * diff)
        else:
            diff = h[0] - (h[1] + 1) // 2
            h[0] = int(h[0] - (2.0 * diff + 1))
    if y < 0.0:
        h[0] = h[0] - (2 * h[1] + 1) // 2
        h[1] = -h[1]


def _hex2d_to_coord_ijk(x, y):
    a1 = abs(x)
    a2 = abs(y)
    x2 = a2 * M_RSIN60
    x1 = a1 + x2 / 2.0
    m1 = int(x1)
    m2 = int(x2)
    i, j = _hex2d_round(x1 - m1, x2 - m2, m1, m2)
    h = [i, j, 0]
    _hex2d_fold(h, x, y)
    _ijk_normalize(h)
    return h


def _unit_ijk_to_digit(ijk):
    c = _ijk_normalize(list(ijk))
    for d in range(CENTER_DIGIT, NUM_DIGITS):
        if c[0] == UNIT_VECS[d][0] and c[1] == UNIT_VECS[d][1] and c[2] == UNIT_VECS[d][2]:
            return d
    return INVALID_DIGIT


def _up_ap7(ijk):
    i = ijk[0] - ijk[2]
    j = ijk[1] - ijk[2]
    ijk[0] = _lround((3 * i - j) * M_ONESEVENTH)
    ijk[1] = _lround((i + 2 * j) * M_ONESEVENTH)
    ijk[2] = 0
    return _ijk_normalize(ijk)


def _up_ap7r(ijk):
    i = ijk[0] - ijk[2]
    j = ijk[1] - ijk[2]
    ijk[0] = _lround((2 * i + j) * M_ONESEVENTH)
    ijk[1] = _lround((3 * j - i) * M_ONESEVENTH)
    ijk[2] = 0
    return _ijk_normalize(ijk)


def _down_ap7(ijk):
    i = _ijk_scale([3, 0, 1], ijk[0])
    j = _ijk_scale([1, 3, 0], ijk[1])
    k = _ijk_scale([0, 1, 3], ijk[2])
    res = _ijk_add(_ijk_add(i, j), k)
    ijk[:] = _ijk_normalize(res)
    return ijk


def _down_ap7r(ijk):
    i = _ijk_scale([3, 1, 0], ijk[0])
    j = _ijk_scale([0, 3, 1], ijk[1])
    k = _ijk_scale([1, 0, 3], ijk[2])
    res = _ijk_add(_ijk_add(i, j), k)
    ijk[:] = _ijk_normalize(res)
    return ijk


def _down_ap3(ijk):
    i = _ijk_scale([2, 0, 1], ijk[0])
    j = _ijk_scale([1, 2, 0], ijk[1])
    k = _ijk_scale([0, 1, 2], ijk[2])
    ijk[:] = _ijk_normalize(_ijk_add(_ijk_add(i, j), k))
    return ijk


def _down_ap3r(ijk):
    i = _ijk_scale([2, 1, 0], ijk[0])
    j = _ijk_scale([0, 2, 1], ijk[1])
    k = _ijk_scale([1, 0, 2], ijk[2])
    ijk[:] = _ijk_normalize(_ijk_add(_ijk_add(i, j), k))
    return ijk


def _neighbor(ijk, digit):
    if CENTER_DIGIT < digit < NUM_DIGITS:
        ijk[:] = _ijk_normalize(_ijk_add(ijk, list(UNIT_VECS[digit])))
    return ijk


def _ijk_rotate60ccw(ijk):
    i = _ijk_scale([1, 1, 0], ijk[0])
    j = _ijk_scale([0, 1, 1], ijk[1])
    k = _ijk_scale([1, 0, 1], ijk[2])
    ijk[:] = _ijk_normalize(_ijk_add(_ijk_add(i, j), k))
    return ijk


def _ijk_rotate60cw(ijk):
    i = _ijk_scale([1, 0, 1], ijk[0])
    j = _ijk_scale([1, 1, 0], ijk[1])
    k = _ijk_scale([0, 1, 1], ijk[2])
    ijk[:] = _ijk_normalize(_ijk_add(_ijk_add(i, j), k))
    return ijk


_ROTATE60CCW = {1: 5, 5: 4, 4: 6, 6: 2, 2: 3, 3: 1, 0: 0}
_ROTATE60CW = {1: 3, 3: 2, 2: 6, 6: 4, 4: 5, 5: 1, 0: 0}


def _rotate60ccw(digit):
    return _ROTATE60CCW[digit]


def _rotate60cw(digit):
    return _ROTATE60CW[digit]


# --- 3D vector math (tuples) ---


def _v3_lincomb(a, v1, b, v2):
    return (a * v1[0] + b * v2[0], a * v1[1] + b * v2[1], a * v1[2] + b * v2[2])


def _v3_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _v3_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v3_normalize(v):
    norm = math.sqrt(_v3_dot(v, v))
    s = 1.0 / norm if norm > 0.0 else 0.0
    return (v[0] * s, v[1] * s, v[2] * s)


def _v3_dist_sq(a, b):
    d = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    return _v3_dot(d, d)


def _latlng_to_vec3(lat, lng):
    r = math.cos(lat)
    return (math.cos(lng) * r, math.sin(lng) * r, math.sin(lat))


def _vec3_to_latlng(v):
    return (math.asin(v[2]), math.atan2(v[1], v[0]))


def _pos_angle_rads(rads):
    tmp = rads + M_2PI if rads < 0.0 else rads
    if rads >= M_2PI:
        tmp -= M_2PI
    return tmp


def _v2d_mag(x, y):
    return math.sqrt(x * x + y * y)


def _v2d_intersect(p0, p1, p2, p3):
    s1x, s1y = p1[0] - p0[0], p1[1] - p0[1]
    s2x, s2y = p3[0] - p2[0], p3[1] - p2[1]
    t = (s2x * (p0[1] - p2[1]) - s2y * (p0[0] - p2[0])) / (-s2x * s1y + s1x * s2y)
    return (p0[0] + t * s1x, p0[1] + t * s1y)


def _v2d_almost_equals(a, b):
    return abs(a[0] - b[0]) < FLT_EPSILON and abs(a[1] - b[1]) < FLT_EPSILON


# --- H3 index bit layout ---


def _get_field(h, offset, width):
    return (h >> offset) & ((1 << width) - 1)


def _set_field(h, val, offset, width):
    mask = (1 << width) - 1
    return (h & ~(mask << offset)) | ((val & mask) << offset)


def _get_resolution(h):
    return _get_field(h, 52, 4)


def _get_base_cell(h):
    return _get_field(h, 45, 7)


def _get_index_digit(h, r):
    return _get_field(h, (MAX_H3_RES - r) * 3, 3)


def _set_index_digit(h, r, v):
    return _set_field(h, v, (MAX_H3_RES - r) * 3, 3)


def _h3_leading_nonzero_digit(h):
    for r in range(1, _get_resolution(h) + 1):
        d = _get_index_digit(h, r)
        if d:
            return d
    return CENTER_DIGIT


def _h3_rotate60ccw(h):
    for r in range(1, _get_resolution(h) + 1):
        h = _set_index_digit(h, r, _rotate60ccw(_get_index_digit(h, r)))
    return h


def _h3_rotate60cw(h):
    for r in range(1, _get_resolution(h) + 1):
        h = _set_index_digit(h, r, _rotate60cw(_get_index_digit(h, r)))
    return h


def _h3_rotate_pent60ccw(h):
    found = False
    for r in range(1, _get_resolution(h) + 1):
        h = _set_index_digit(h, r, _rotate60ccw(_get_index_digit(h, r)))
        if not found and _get_index_digit(h, r) != 0:
            found = True
            if _h3_leading_nonzero_digit(h) == K_AXES_DIGIT:
                h = _h3_rotate60ccw(h)
    return h


# --- base cell helpers ---


def _is_base_cell_pentagon(bc):
    if bc < 0 or bc >= NUM_BASE_CELLS:
        return False
    return BASE_CELL_DATA[bc][4] == 1


def _base_cell_is_cw_offset(bc, test_face):
    return BASE_CELL_DATA[bc][5] == test_face or BASE_CELL_DATA[bc][6] == test_face


def _base_cell_home_fijk(bc):
    d = BASE_CELL_DATA[bc]
    return d[0], [d[1], d[2], d[3]]


def _face_ijk_to_base_cell(face, coord):
    return FACE_IJK_BASE_CELLS[face][coord[0]][coord[1]][coord[2]][0]


def _face_ijk_to_base_cell_ccw_rot60(face, coord):
    return FACE_IJK_BASE_CELLS[face][coord[0]][coord[1]][coord[2]][1]


# --- faceijk projection ---


def _vec3_to_closest_face(v):
    face = 0
    sqd = 5.0
    for f in range(20):
        t = _v3_dist_sq(FACE_CENTER_POINT[f], v)
        if t < sqd:
            face = f
            sqd = t
    return face, sqd


def _vec3_tangent_basis(p):
    north_pole = (0.0, 0.0, 1.0)
    north = _v3_normalize(_v3_lincomb(1.0, north_pole, -_v3_dot(north_pole, p), p))
    east = _v3_cross(north, p)
    return north, east


def _vec3_azimuth_rads(p1, p2):
    north, east = _vec3_tangent_basis(p1)
    p2proj = _v3_normalize(_v3_lincomb(1.0, p2, -_v3_dot(p2, p1), p1))
    return math.atan2(_v3_dot(p2proj, east), _v3_dot(p2proj, north))


def _vec3_to_hex2d(p, res):
    face, sqd = _vec3_to_closest_face(p)
    r = math.acos(1 - sqd * 0.5)
    if r < EPSILON:
        return face, 0.0, 0.0
    theta = _pos_angle_rads(
        FACE_AXES_AZ_RADS_CII[face][0] - _pos_angle_rads(_vec3_azimuth_rads(FACE_CENTER_POINT[face], p))
    )
    if is_class_iii(res):
        theta = _pos_angle_rads(theta - M_AP7_ROT_RADS)
    r = math.tan(r)
    r *= INV_RES0_U_GNOMONIC
    for _ in range(res):
        r *= M_SQRT7
    return face, r * math.cos(theta), r * math.sin(theta)


def _vec3_to_faceijk(p, res):
    face, x, y = _vec3_to_hex2d(p, res)
    return face, _hex2d_to_coord_ijk(x, y)


def _hex2d_to_vec3(x, y, face, res, substrate):
    r = _v2d_mag(x, y)
    if r < EPSILON:
        return FACE_CENTER_POINT[face]
    theta = math.atan2(y, x)
    for _ in range(res):
        r *= M_RSQRT7
    if substrate:
        r *= M_ONETHIRD
        if is_class_iii(res):
            r *= M_RSQRT7
    r *= RES0_U_GNOMONIC
    r = math.atan(r)
    if not substrate and is_class_iii(res):
        theta = _pos_angle_rads(theta + M_AP7_ROT_RADS)
    theta = _pos_angle_rads(FACE_AXES_AZ_RADS_CII[face][0] - theta)
    north, east = _vec3_tangent_basis(FACE_CENTER_POINT[face])
    direction = _v3_lincomb(math.cos(theta), north, math.sin(theta), east)
    return _v3_normalize(_v3_lincomb(math.cos(r), FACE_CENTER_POINT[face], math.sin(r), direction))


def _overage_orient(face, ijk, max_dim, pent_leading4):
    """Pick the adjacent-face orientation for an overage point and apply pentagon-gap rotation (in place)."""
    if ijk[2] > 0:
        if ijk[1] > 0:
            return FACE_NEIGHBORS[face][JK]
        orient = FACE_NEIGHBORS[face][KI]
        if pent_leading4:
            origin = [max_dim, 0, 0]
            tmp = _ijk_sub(ijk, origin)
            _ijk_rotate60cw(tmp)
            ijk[:] = _ijk_add(tmp, origin)
        return orient
    return FACE_NEIGHBORS[face][IJ]


def _adjust_overage_class_ii(face, ijk, res, pent_leading4, substrate):
    overage = NO_OVERAGE
    max_dim = MAX_DIM_BY_CII_RES[res]
    if substrate:
        max_dim *= 3
    total = ijk[0] + ijk[1] + ijk[2]
    if substrate and total == max_dim:
        return face, FACE_EDGE
    if total <= max_dim:
        return face, overage
    overage = NEW_FACE
    orient = _overage_orient(face, ijk, max_dim, pent_leading4)
    face = orient[0]
    for _ in range(orient[4]):
        _ijk_rotate60ccw(ijk)
    unit_scale = UNIT_SCALE_BY_CII_RES[res] * (3 if substrate else 1)
    ijk[:] = _ijk_normalize(_ijk_add(ijk, _ijk_scale([orient[1], orient[2], orient[3]], unit_scale)))
    if substrate and ijk[0] + ijk[1] + ijk[2] == max_dim:
        overage = FACE_EDGE
    return face, overage


def _adjust_pent_vert_overage(face, ijk, res):
    overage = NEW_FACE
    while overage == NEW_FACE:
        face, overage = _adjust_overage_class_ii(face, ijk, res, 0, 1)
    return face, overage


_VERTS_CII = ([2, 1, 0], [1, 2, 0], [0, 2, 1], [0, 1, 2], [1, 0, 2], [2, 0, 1])
_VERTS_CIII = ([5, 4, 0], [1, 5, 0], [0, 5, 4], [0, 1, 5], [4, 0, 5], [5, 0, 1])


def _faceijk_to_verts(face, coord, res, n_verts):
    verts = _VERTS_CIII if is_class_iii(res) else _VERTS_CII
    _down_ap3(coord)
    _down_ap3r(coord)
    adj_res = res
    if is_class_iii(res):
        _down_ap7r(coord)
        adj_res += 1
    out = []
    for v in range(n_verts):
        vc = _ijk_normalize(_ijk_add(coord, verts[v]))
        out.append([face, vc])
    return adj_res, out


def _icosa_edge_endpoints(max_dim, d):
    """Return the two icosa-face-edge endpoints (hex2d) for adjacent-face direction ``d``."""
    v0 = (3.0 * max_dim, 0.0)
    v1 = (-1.5 * max_dim, 3.0 * M_SQRT3_2 * max_dim)
    v2 = (-1.5 * max_dim, -3.0 * M_SQRT3_2 * max_dim)
    if d == IJ:
        return v0, v1
    if d == JK:
        return v1, v2
    return v2, v0


def _project_coord(coord, face, adj_res):
    """Project a substrate hex2d cell coordinate to a ``(lat, lng)`` radian pair."""
    vec = _ijk_to_hex2d(coord)
    return _vec3_to_latlng(_hex2d_to_vec3(vec[0], vec[1], face, adj_res, 1))


def _faceijk_hex_boundary(face, coord, res):
    adj_res, fijk_verts = _faceijk_to_verts(face, list(coord), res, NUM_HEX_VERTS)
    center_face = face
    out = []
    last_face = -1
    last_overage = NO_OVERAGE
    for vert in range(NUM_HEX_VERTS + 1):
        v = vert % NUM_HEX_VERTS
        vface, vcoord = fijk_verts[v][0], list(fijk_verts[v][1])
        vface, overage = _adjust_overage_class_ii(vface, vcoord, adj_res, 0, 1)
        if is_class_iii(res) and vert > 0 and vface != last_face and last_overage != FACE_EDGE:
            orig2d0 = _ijk_to_hex2d(fijk_verts[(v + 5) % NUM_HEX_VERTS][1])
            orig2d1 = _ijk_to_hex2d(fijk_verts[v][1])
            face2 = vface if last_face == center_face else last_face
            edge0, edge1 = _icosa_edge_endpoints(MAX_DIM_BY_CII_RES[adj_res], ADJACENT_FACE_DIR[center_face][face2])
            inter = _v2d_intersect(orig2d0, orig2d1, edge0, edge1)
            if not (_v2d_almost_equals(orig2d0, inter) or _v2d_almost_equals(orig2d1, inter)):
                out.append(_vec3_to_latlng(_hex2d_to_vec3(inter[0], inter[1], center_face, adj_res, 1)))
        if vert < NUM_HEX_VERTS:
            out.append(_project_coord(vcoord, vface, adj_res))
        last_face = vface
        last_overage = overage
    return out


def _pent_edge_point(last_fijk, vface, vcoord, adj_res):
    """Compute the icosa-edge-crossing vertex between two pentagon boundary vertices."""
    orig2d0 = _ijk_to_hex2d(last_fijk[1])
    tmp_coord = list(vcoord)
    orient = FACE_NEIGHBORS[vface][ADJACENT_FACE_DIR[vface][last_fijk[0]]]
    tmp_face = orient[0]
    for _ in range(orient[4]):
        _ijk_rotate60ccw(tmp_coord)
    trans = _ijk_scale([orient[1], orient[2], orient[3]], UNIT_SCALE_BY_CII_RES[adj_res] * 3)
    tmp_coord = _ijk_normalize(_ijk_add(tmp_coord, trans))
    orig2d1 = _ijk_to_hex2d(tmp_coord)
    edge0, edge1 = _icosa_edge_endpoints(MAX_DIM_BY_CII_RES[adj_res], ADJACENT_FACE_DIR[tmp_face][vface])
    inter = _v2d_intersect(orig2d0, orig2d1, edge0, edge1)
    return _vec3_to_latlng(_hex2d_to_vec3(inter[0], inter[1], tmp_face, adj_res, 1))


def _faceijk_pent_boundary(face, coord, res):
    adj_res, fijk_verts = _faceijk_to_verts(face, list(coord), res, NUM_PENT_VERTS)
    out = []
    last_fijk = [0, [0, 0, 0]]
    for vert in range(NUM_PENT_VERTS + 1):
        v = vert % NUM_PENT_VERTS
        vface, vcoord = fijk_verts[v][0], list(fijk_verts[v][1])
        vface, _ = _adjust_pent_vert_overage(vface, vcoord, adj_res)
        if is_class_iii(res) and vert > 0:
            out.append(_pent_edge_point(last_fijk, vface, vcoord, adj_res))
        if vert < NUM_PENT_VERTS:
            out.append(_project_coord(vcoord, vface, adj_res))
        last_fijk = [vface, vcoord]
    return out


def _faceijk_to_cell_boundary(face, coord, res, is_pentagon):
    if is_pentagon:
        return _faceijk_pent_boundary(face, coord, res)
    return _faceijk_hex_boundary(face, coord, res)


# --- faceijk <-> H3 index ---


def _faceijk_set_digits(h, ijk, res):
    """Walk the cell up the aperture-7 hierarchy, setting each resolution's index digit (ijk left at res 0)."""
    for r in range(res - 1, -1, -1):
        last = list(ijk)
        if is_class_iii(r + 1):
            _up_ap7(ijk)
            center = list(ijk)
            _down_ap7(center)
        else:
            _up_ap7r(ijk)
            center = list(ijk)
            _down_ap7r(center)
        diff = _ijk_normalize(_ijk_sub(last, center))
        h = _set_index_digit(h, r + 1, _unit_ijk_to_digit(diff))
    return h


def _apply_base_cell_rotation(h, base_cell, face, num_rots):
    """Apply the base-cell ccw rotations, with the pentagon leading-K and pent-rotation adjustments."""
    if _is_base_cell_pentagon(base_cell):
        if _h3_leading_nonzero_digit(h) == K_AXES_DIGIT:
            if _base_cell_is_cw_offset(base_cell, face):
                h = _h3_rotate60cw(h)
            else:
                h = _h3_rotate60ccw(h)
        for _ in range(num_rots):
            h = _h3_rotate_pent60ccw(h)
    else:
        for _ in range(num_rots):
            h = _h3_rotate60ccw(h)
    return h


def _faceijk_to_h3(face, coord, res):
    h = H3_INIT
    h = _set_field(h, H3_CELL_MODE, 59, 4)
    h = _set_field(h, res, 52, 4)
    if res == 0:
        return _set_field(h, _face_ijk_to_base_cell(face, coord), 45, 7)
    ijk = list(coord)
    h = _faceijk_set_digits(h, ijk, res)
    base_cell = _face_ijk_to_base_cell(face, ijk)
    h = _set_field(h, base_cell, 45, 7)
    num_rots = _face_ijk_to_base_cell_ccw_rot60(face, ijk)
    return _apply_base_cell_rotation(h, base_cell, face, num_rots)


def _h3_faceijk_overage(face, coord, h, base_cell, res, orig):
    """Apply the class-II overage adjustment that relocates a cell onto its correct icosa face."""
    adj_res = res
    if is_class_iii(res):
        _down_ap7r(coord)
        adj_res += 1
    pent_leading4 = _is_base_cell_pentagon(base_cell) and _h3_leading_nonzero_digit(h) == 4
    face, overage = _adjust_overage_class_ii(face, coord, adj_res, 1 if pent_leading4 else 0, 0)
    if overage == NO_OVERAGE:
        return face, (list(orig) if adj_res != res else coord)
    if _is_base_cell_pentagon(base_cell):
        ov = overage
        while ov != NO_OVERAGE:
            face, ov = _adjust_overage_class_ii(face, coord, adj_res, 0, 0)
    if adj_res != res:
        _up_ap7r(coord)
    return face, coord


def _h3_to_faceijk(h):
    base_cell = _get_base_cell(h)
    if _is_base_cell_pentagon(base_cell) and _h3_leading_nonzero_digit(h) == 5:
        h = _h3_rotate60cw(h)
    face, coord = _base_cell_home_fijk(base_cell)
    coord = list(coord)
    res = _get_resolution(h)
    possible_overage = not (
        not _is_base_cell_pentagon(base_cell) and (res == 0 or coord == [0, 0, 0])
    )
    for r in range(1, res + 1):
        if is_class_iii(r):
            _down_ap7(coord)
        else:
            _down_ap7r(coord)
        _neighbor(coord, _get_index_digit(h, r))
    if not possible_overage:
        return face, coord
    return _h3_faceijk_overage(face, coord, h, base_cell, res, list(coord))


# --- public API ---


def latlng_to_cell(lat_deg: float, lng_deg: float, res: int) -> str:
    """Return the H3 cell index (lowercase hex string) containing the lat/lng point.

    Args:
        lat_deg: Latitude in decimal degrees.
        lng_deg: Longitude in decimal degrees.
        res: H3 resolution, 0-15.

    Returns:
        str: The H3 cell index as a lowercase hex string (e.g. ``"89283082803ffff"``).
    """
    if res < 0 or res > MAX_H3_RES:
        raise ValueError(f"H3 resolution must be 0-15, got {res}")
    v = _latlng_to_vec3(lat_deg * M_PI_180, lng_deg * M_PI_180)
    face, coord = _vec3_to_faceijk(v, res)
    return format(_faceijk_to_h3(face, coord, res), "x")


def cell_to_boundary(cell: str) -> list[tuple[float, float]]:
    """Return the boundary vertices of an H3 cell as ``(lat, lng)`` pairs in degrees.

    Args:
        cell: H3 cell index as a hex string.

    Returns:
        list[tuple[float, float]]: The hexagon (or pentagon) vertices in degrees, counter-clockwise.
    """
    h = int(cell, 16)
    base_cell = _get_base_cell(h)
    is_pent = _is_base_cell_pentagon(base_cell) and _h3_leading_nonzero_digit(h) == 0
    face, coord = _h3_to_faceijk(h)
    verts = _faceijk_to_cell_boundary(face, coord, _get_resolution(h), is_pent)
    return [(lat * M_180_PI, lng * M_180_PI) for (lat, lng) in verts]
