

import os, math, zipfile, warnings
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

try:
    from google.colab import files
    IN_COLAB = True
except Exception:
    IN_COLAB = False

np.random.seed(0)

# -------------------------
# CONFIG
# -------------------------
OUTDIR = "paper_outputs"
os.makedirs(OUTDIR, exist_ok=True)

SUBDIV_DEFAULT = 2
JAM_M_DEFAULT  = 50.0

EDGE_TMAX      = 60.0
EDGE_NT        = 90
EDGE_K         = 40

Z_CUT_CAP  = 0.15   # ex1 cap: z > Z_CUT_CAP
Z_BAND     = 0.35   # ex2 band: |z| < Z_BAND

C1 = np.array([0.65, 0.00, 0.76]); C1 /= np.linalg.norm(C1)
C2 = np.array([0.00, 0.65, 0.76]); C2 /= np.linalg.norm(C2)
R_HINT_EX3 = 0.225

G_FLOOR = 1e-6

M_LIST_RESOLVENT = [10.0, 20.0, 40.0, 50.0, 100.0]
M_LIST_LEAKAGE   = [10.0, 20.0, 40.0, 50.0, 100.0]
M_LIST_EIGLOC    = [10.0, 20.0, 40.0, 50.0, 100.0]
DISORDER_FRAC    = 0.20
FLUX_N_THETA     = 25
FLUX_K           = 18
PL_SCALING_LEVELS = (1, 2, 3, 4)

# -------------------------
# Output helpers
# -------------------------
def savefig(name, dpi=220):
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path

def save_placeholder_png(outname, message):
    plt.figure(figsize=(7.8, 3.6))
    plt.axis("off")
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return savefig(outname)

def write_zip_bundle(all_files, zip_name="figures_and_tables.zip"):
    zip_path = os.path.join(OUTDIR, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(set(all_files)):
            if os.path.exists(fp):
                zf.write(fp, arcname=os.path.basename(fp))
    return zip_path

def robust_half_gap(evals):
    evals = np.asarray(evals)
    g_raw = float(np.min(np.abs(evals)))
    g = 0.0 if g_raw < G_FLOOR else g_raw
    return g_raw, g

def _tex_num(x, fixed=4, sci_lo=1e-3, sci_hi=1e3):
    if x is None:
        return r"\text{--}"
    try:
        x = float(x)
    except Exception:
        return r"\text{--}"
    if not np.isfinite(x):
        return r"\text{--}"
    ax = abs(x)
    if ax == 0.0:
        return "0"
    if (ax >= sci_lo) and (ax < sci_hi):
        return f"{x:.{fixed}f}"
    exp = int(np.floor(np.log10(ax)))
    mant = x / (10.0 ** exp)
    return rf"{mant:.3f}\times 10^{{{exp}}}"

# -------------------------
# Icosphere generation
# -------------------------
def icosahedron():
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        (-1,  t,  0),( 1,  t,  0),(-1, -t,  0),( 1, -t,  0),
        ( 0, -1,  t),( 0,  1,  t),( 0, -1, -t),( 0,  1, -t),
        ( t,  0, -1),( t,  0,  1),(-t,  0, -1),(-t,  0,  1),
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ], dtype=int)
    return verts, faces

def orient_faces_outward(verts, faces):
    faces = faces.copy()
    for i,(a,b,c) in enumerate(faces):
        va, vb, vc = verts[a], verts[b], verts[c]
        n = np.cross(vb-va, vc-va)
        if np.dot(n, va) < 0:
            faces[i] = (a,c,b)
    return faces

def subdivide_icosphere(verts, faces, subdivisions):
    verts_list = verts.tolist()
    faces_list = faces.tolist()
    for _ in range(subdivisions):
        mid_cache = {}
        def midpoint(i, j):
            key = (i, j) if i < j else (j, i)
            if key in mid_cache:
                return mid_cache[key]
            vi = np.array(verts_list[i]); vj = np.array(verts_list[j])
            vm = (vi + vj) / 2.0
            vm = vm / np.linalg.norm(vm)
            verts_list.append(vm.tolist())
            idx = len(verts_list) - 1
            mid_cache[key] = idx
            return idx

        new_faces = []
        for (a,b,c) in faces_list:
            ab = midpoint(a,b); bc = midpoint(b,c); ca = midpoint(c,a)
            new_faces += [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        verts_arr = np.array(verts_list, dtype=float)
        faces_arr = orient_faces_outward(verts_arr, np.array(new_faces, dtype=int))
        faces_list = faces_arr.tolist()

    verts_arr = np.array(verts_list, dtype=float)
    faces_arr = orient_faces_outward(verts_arr, np.array(faces_list, dtype=int))
    return verts_arr, faces_arr

# -------------------------
# Subcomplex selection
# -------------------------
def select_subcomplex_by_vertex_predicate(verts, faces, pred_v):
    keep_face = []
    for (a,b,c) in faces:
        keep_face.append(pred_v(verts[a]) and pred_v(verts[b]) and pred_v(verts[c]))
    faces_sel = faces[np.array(keep_face, dtype=bool)]
    if faces_sel.shape[0] == 0:
        raise ValueError("No faces selected; relax predicate.")
    return faces_sel

def faces_largest_component(faces):
    edge_to_faces = {}
    for i, (a,b,c) in enumerate(faces):
        for x,y in [(a,b),(b,c),(c,a)]:
            e = (x,y) if x<y else (y,x)
            edge_to_faces.setdefault(e, []).append(i)
    adj = [set() for _ in range(len(faces))]
    for flist in edge_to_faces.values():
        if len(flist) >= 2:
            u = flist[0]
            for v in flist[1:]:
                adj[u].add(v); adj[v].add(u)
    seen = set()
    comps = []
    for i in range(len(faces)):
        if i in seen: continue
        stack = [i]; seen.add(i); comp=[i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v); comp.append(v)
        comps.append(comp)
    largest = max(comps, key=len)
    return faces[np.array(largest, dtype=int)]

def reindex_submesh(verts, faces):
    used = np.unique(faces.reshape(-1))
    mapping = {old:i for i,old in enumerate(used)}
    verts_new = verts[used]
    faces_new = np.vectorize(mapping.get)(faces)
    return verts_new, faces_new

# -------------------------
# Edges / boundary cycles
# -------------------------
def compute_edges_from_faces(faces):
    edge_to_id = {}
    u=[]; v=[]
    face_edge_ids=[]
    for (a,b,c) in faces:
        eids=[]
        for x,y in [(a,b),(b,c),(c,a)]:
            e = (x,y) if x<y else (y,x)
            if e not in edge_to_id:
                edge_to_id[e]=len(u)
                u.append(e[0]); v.append(e[1])
            eids.append(edge_to_id[e])
        face_edge_ids.append(eids)
    return np.array(u,int), np.array(v,int), edge_to_id, np.array(face_edge_ids,int)

def boundary_data(faces):
    u,v,edge_to_id,face_edge_ids = compute_edges_from_faces(faces)
    inc = np.zeros(len(u), dtype=int)
    incident_face = -np.ones(len(u), dtype=int)
    for fid, eids in enumerate(face_edge_ids):
        for eid in eids:
            inc[eid]+=1
            incident_face[eid]=fid if incident_face[eid]==-1 else incident_face[eid]
    boundary_eids = np.where(inc==1)[0]
    bedges = [(int(u[eid]), int(v[eid]), int(eid)) for eid in boundary_eids]
    return u,v,edge_to_id,face_edge_ids,inc,incident_face,bedges

def oriented_boundary_cycles(faces):
    u,v,edge_to_id,face_edge_ids,inc,incident_face,bedges = boundary_data(faces)
    nxt = {}
    indeg = {}
    for (a,b,eid) in bedges:
        fid = incident_face[eid]
        x,y,z = map(int, faces[fid])
        oriented_edges = [(y,z),(z,x),(x,y)]
        if (a,b) in oriented_edges:
            s,t = a,b
        elif (b,a) in oriented_edges:
            s,t = b,a
        else:
            raise RuntimeError("Boundary edge not found in oriented face boundary.")
        if s in nxt:
            raise ValueError("Non-manifold boundary: multiple outgoing boundary edges.")
        nxt[s]=t
        indeg[t]=indeg.get(t,0)+1
    for s,t in nxt.items():
        if indeg.get(s,0)!=1:
            raise ValueError("Non-manifold boundary: indegree != 1 at a boundary vertex.")
    cycles=[]
    seen=set()
    for start in list(nxt.keys()):
        if start in seen: continue
        cyc=[start]; seen.add(start)
        cur=nxt[start]
        while cur!=start:
            cyc.append(cur); seen.add(cur)
            cur=nxt[cur]
        cycles.append(cyc)
    return cycles

def assert_manifold_with_boundary(verts, faces, name="(mesh)"):
    u,v,edge_to_id,face_edge_ids,inc,incident_face,bedges = boundary_data(faces)
    if np.any((inc!=1)&(inc!=2)):
        bad = np.where((inc!=1)&(inc!=2))[0][:10]
        raise ValueError(f"{name}: edges with incidence not in {{1,2}}. Example eids={bad}.")
    cycles = oriented_boundary_cycles(faces)
    bverts = set().union(*[set(c) for c in cycles]) if cycles else set()
    boundary_edge_count = int(np.sum(inc==1))
    boundary_vertex_count = len(bverts)
    if boundary_edge_count != boundary_vertex_count:
        raise ValueError(
            f"{name}: boundary edges != boundary vertices ({boundary_edge_count} vs {boundary_vertex_count})."
        )
    return cycles, boundary_vertex_count, boundary_edge_count

def assert_closed_triangulated_surface(faces, name="(double)"):
    u,v,edge_to_id,face_edge_ids,inc,_,_ = boundary_data(faces)
    if not np.all(inc==2):
        bad = np.where(inc!=2)[0][:10]
        raise ValueError(f"{name}: not closed triangulated surface (edges with incidence !=2), sample eids={bad}.")

def boundary_vertices_set(faces):
    cycles = oriented_boundary_cycles(faces)
    return set().union(*[set(c) for c in cycles]) if cycles else set()

def boundary_is_induced(faces):
    u,v,edge_to_id,face_edge_ids,inc,_,_ = boundary_data(faces)
    bverts = boundary_vertices_set(faces)
    # violation: an interior edge (inc==2) whose endpoints are both boundary vertices
    for eid,(a,b) in enumerate(zip(u,v)):
        if inc[eid]==2 and (int(a) in bverts) and (int(b) in bverts):
            return False
    return True

# -------------------------
# Barycentric subdivision (sphere-projected)
# -------------------------
def barycentric_subdivide_once(verts, faces, project_to_sphere=True):
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)

    verts_list = verts.tolist()
    edge_mid = {}
    face_cent = []

    def add_vertex(p):
        p = np.asarray(p, float)
        if project_to_sphere:
            n = np.linalg.norm(p)
            if n > 0:
                p = p / n
        verts_list.append(p.tolist())
        return len(verts_list)-1

    def mid_idx(i,j):
        key = (i,j) if i<j else (j,i)
        if key in edge_mid:
            return edge_mid[key]
        m = 0.5*(verts[i]+verts[j])
        idx = add_vertex(m)
        edge_mid[key] = idx
        return idx

    new_faces = []
    for (a,b,c) in faces:
        a=int(a); b=int(b); c=int(c)
        mab = mid_idx(a,b)
        mbc = mid_idx(b,c)
        mca = mid_idx(c,a)
        fc  = add_vertex((verts[a]+verts[b]+verts[c])/3.0)

        # 6 triangles around face centroid
        new_faces += [
            (a,  mab, fc),
            (mab, b,  fc),
            (b,  mbc, fc),
            (mbc, c,  fc),
            (c,  mca, fc),
            (mca, a,  fc),
        ]

    verts2 = np.array(verts_list, float)
    faces2 = orient_faces_outward(verts2, np.array(new_faces, int))
    return verts2, faces2

def ensure_induced_boundary(verts, faces, name="patch", max_rounds=2):
    # Always ensure manifold-with-boundary first
    assert_manifold_with_boundary(verts, faces, name=name)

    if boundary_is_induced(faces):
        return verts, faces, 0

    v, f = verts, faces
    rounds = 0
    while rounds < max_rounds and (not boundary_is_induced(f)):
        v, f = barycentric_subdivide_once(v, f, project_to_sphere=True)
        assert_manifold_with_boundary(v, f, name=name+f" (bary{rounds+1})")
        rounds += 1

    if not boundary_is_induced(f):
        raise ValueError(f"{name}: could not enforce induced boundary after {max_rounds} barycentric rounds.")
    return v, f, rounds

# -------------------------
# Example builders (paper geometry)
# -------------------------
def build_ex1_cap(vertsS, facesS, z_cut=Z_CUT_CAP):
    def pred_v(v): return v[2] > z_cut
    faces_sel = select_subcomplex_by_vertex_predicate(vertsS, facesS, pred_v)
    faces_sel = faces_largest_component(faces_sel)
    v_patch, f_patch = reindex_submesh(vertsS, faces_sel)
    f_patch = orient_faces_outward(v_patch, f_patch)
    cycles, bv, be = assert_manifold_with_boundary(v_patch, f_patch, name="ex1")
    if len(cycles) != 1:
        raise ValueError(f"ex1: expected 1 boundary, got {len(cycles)}")
    v_patch, f_patch, rounds = ensure_induced_boundary(v_patch, f_patch, name="ex1")
    cycles, bv, be = assert_manifold_with_boundary(v_patch, f_patch, name="ex1(final)")
    return v_patch, f_patch, cycles, bv, be, rounds

def build_ex2_band(vertsS, facesS, z_band=Z_BAND):
    def pred_v(v): return abs(v[2]) < z_band
    faces_sel = select_subcomplex_by_vertex_predicate(vertsS, facesS, pred_v)
    faces_sel = faces_largest_component(faces_sel)
    v_patch, f_patch = reindex_submesh(vertsS, faces_sel)
    f_patch = orient_faces_outward(v_patch, f_patch)
    cycles, bv, be = assert_manifold_with_boundary(v_patch, f_patch, name="ex2")
    if len(cycles) != 2:
        raise ValueError(f"ex2: expected 2 boundaries, got {len(cycles)}")
    v_patch, f_patch, rounds = ensure_induced_boundary(v_patch, f_patch, name="ex2")
    cycles, bv, be = assert_manifold_with_boundary(v_patch, f_patch, name="ex2(final)")
    return v_patch, f_patch, cycles, bv, be, rounds

def build_open_patch_from_cap_with_holes(vertsS, facesS, z_cut, hole_centers, r_hint, expected_num_boundaries, tag):
    hole_centers = [np.asarray(c, float)/np.linalg.norm(c) for c in hole_centers]
    r_min, r_max = 0.05, 0.45
    grid = np.linspace(r_min, r_max, 81)
    focus = np.linspace(max(r_min, 0.6*r_hint), min(r_max, 1.4*r_hint), 31)
    r_candidates = np.unique(np.concatenate([focus, grid]))
    r_candidates = sorted(r_candidates, key=lambda r: abs(r - r_hint))

    last_err = None
    last_seen = None

    for r in r_candidates:
        def pred_v(v):
            if v[2] <= z_cut: return False
            for c in hole_centers:
                if np.linalg.norm(v - c) <= r: return False
            return True
        try:
            faces_sel = select_subcomplex_by_vertex_predicate(vertsS, facesS, pred_v)
            faces_sel = faces_largest_component(faces_sel)
            v_patch, f_patch = reindex_submesh(vertsS, faces_sel)
            f_patch = orient_faces_outward(v_patch, f_patch)

            cycles, bv, be = assert_manifold_with_boundary(v_patch, f_patch, name=tag)
            sizes = sorted([len(c) for c in cycles], reverse=True)
            last_seen = (len(cycles), sizes, bv, be, v_patch.shape[0], f_patch.shape[0])

            if len(cycles) == expected_num_boundaries:
                v_patch, f_patch, rounds = ensure_induced_boundary(v_patch, f_patch, name=tag)
                cycles, bv, be = assert_manifold_with_boundary(v_patch, f_patch, name=tag+"(final)")
                if len(cycles) == expected_num_boundaries:
                    return v_patch, f_patch, cycles, bv, be, float(r), rounds
        except Exception as e:
            last_err = str(e)

    raise ValueError(
        f"{tag}: Could not realize exactly {expected_num_boundaries} boundary components.\n"
        f"  last_seen={last_seen}\n"
        f"  last_err={last_err}"
    )

def build_ex3_two_holes(vertsS, facesS, z_cut=Z_CUT_CAP, centers=(C1,C2), r_hint=R_HINT_EX3):
    v_patch, f_patch, cycles, bv, be, r_used, rounds = build_open_patch_from_cap_with_holes(
        vertsS, facesS, z_cut=z_cut,
        hole_centers=list(centers),
        r_hint=r_hint,
        expected_num_boundaries=3,
        tag="ex3"
    )
    return v_patch, f_patch, cycles, bv, be, r_used, rounds

# -------------------------
# Doubling (now safe because boundary is induced)
# -------------------------
def build_double(verts, faces):
    cycles, _, _ = assert_manifold_with_boundary(verts, faces, name="open patch")
    boundary_vertices = set().union(*[set(c) for c in cycles])

    nV0 = verts.shape[0]
    verts2 = verts.tolist()
    copy_map = {}
    copy_only_vertices=set()

    for i in range(nV0):
        if i in boundary_vertices:
            copy_map[i]=i
        else:
            copy_map[i]=len(verts2)
            verts2.append(verts[i].tolist())
            copy_only_vertices.add(copy_map[i])

    verts2 = np.array(verts2, dtype=float)

    faces_copy=[]
    for (a,b,c) in faces:
        a2,b2,c2 = copy_map[int(a)], copy_map[int(b)], copy_map[int(c)]
        faces_copy.append((a2, c2, b2))  # reverse orientation for the copy
    faces2 = np.vstack([faces, np.array(faces_copy, dtype=int)])

    # HARD correctness check: double must be closed triangulated surface
    assert_closed_triangulated_surface(faces2, name="D(K)")
    return verts2, faces2, cycles, boundary_vertices, copy_map, copy_only_vertices

def edge_incident_faces(faces, u, v, edge_to_id):
    inc = [[] for _ in range(len(u))]
    for fid, (a,b,c) in enumerate(faces):
        for x,y in [(a,b),(b,c),(c,a)]:
            e = (x,y) if x<y else (y,x)
            inc[edge_to_id[e]].append(fid)
    return inc

# -------------------------
# Build B, P, S, H
# -------------------------
def build_B_P_S(verts, faces):
    nV = verts.shape[0]
    u, v, edge_to_id, face_edge_ids = compute_edges_from_faces(faces)
    nE = len(u)
    nF = faces.shape[0]
    N = nV + nE + nF

    def edge_id_sign(x,y):
        if x<y:
            e=(x,y); s=+1
        else:
            e=(y,x); s=-1
        return edge_to_id[e], s

    # B
    rows=[]; cols=[]; data=[]
    for eid,(a,b) in enumerate(zip(u,v)):
        cols += [nV+eid, nV+eid]
        rows += [b, a]
        data += [1.0, -1.0]
    for fid,(a,b,c) in enumerate(faces):
        fcol = nV+nE+fid
        eid,s = edge_id_sign(b,c); rows.append(nV+eid); cols.append(fcol); data.append( 1.0*s)
        eid,s = edge_id_sign(a,c); rows.append(nV+eid); cols.append(fcol); data.append(-1.0*s)
        eid,s = edge_id_sign(a,b); rows.append(nV+eid); cols.append(fcol); data.append( 1.0*s)
    B = sp.coo_matrix((data,(rows,cols)), shape=(N,N), dtype=float).tocsr()

    # P (cap-product proxy)
    rows=[]; cols=[]; data=[]
    for fid,(a,b,c) in enumerate(faces):
        face_idx = nV+nE+fid
        rows.append(face_idx); cols.append(c); data.append(1.0)   # 0 -> 2
        rows.append(a); cols.append(face_idx); data.append(1.0)   # 2 -> 0
        eid_in, s_in = edge_id_sign(b,c)
        eid_out,s_out= edge_id_sign(a,b)
        rows.append(nV+eid_out); cols.append(nV+eid_in); data.append(float(s_out*s_in)) # 1 -> 1
    P = sp.coo_matrix((data,(rows,cols)), shape=(N,N), dtype=float).tocsr()

    # S from P
    d = np.concatenate([np.ones(nV), -np.ones(nE), np.ones(nF)])
    Ddeg = sp.diags(d, 0, format="csr")
    cdiag = np.concatenate([1j*np.ones(nV), 1j*np.ones(nE), (-1j)*np.ones(nF)])
    C = sp.diags(cdiag, 0, format="csr")
    T = 0.5*(P.getH() + P@Ddeg)
    S = T@C

    return B, P, S, (u,v,edge_to_id), (nV,nE,nF)

def build_B_P_S_relative(verts, faces):
    nV = verts.shape[0]
    u, v, edge_to_id, face_edge_ids, inc, _, _ = boundary_data(faces)
    nE = len(u)
    nF = faces.shape[0]
    N = nV + nE + nF
    boundary_vertices = boundary_vertices_set(faces)
    boundary_edges = set(np.where(inc == 1)[0].astype(int).tolist())

    def edge_id_sign(x, y):
        if x < y:
            e = (x, y); s = +1
        else:
            e = (y, x); s = -1
        return edge_to_id[e], s

    rows=[]; cols=[]; data=[]
    for eid,(a,b) in enumerate(zip(u,v)):
        cols += [nV+eid, nV+eid]
        rows += [b, a]
        data += [1.0, -1.0]
    for fid,(a,b,c) in enumerate(faces):
        fcol = nV+nE+fid
        eid,s = edge_id_sign(b,c); rows.append(nV+eid); cols.append(fcol); data.append( 1.0*s)
        eid,s = edge_id_sign(a,c); rows.append(nV+eid); cols.append(fcol); data.append(-1.0*s)
        eid,s = edge_id_sign(a,b); rows.append(nV+eid); cols.append(fcol); data.append( 1.0*s)
    B = sp.coo_matrix((data,(rows,cols)), shape=(N,N), dtype=float).tocsr()

    rows=[]; cols=[]; data=[]
    for fid,(a,b,c) in enumerate(faces):
        face_idx = nV+nE+fid
        rows.append(face_idx); cols.append(c); data.append(1.0)
        if int(a) not in boundary_vertices:
            rows.append(a); cols.append(face_idx); data.append(1.0)
        eid_in, s_in = edge_id_sign(b,c)
        eid_out, s_out = edge_id_sign(a,b)
        if int(eid_out) not in boundary_edges:
            rows.append(nV+eid_out); cols.append(nV+eid_in); data.append(float(s_out*s_in))
    P = sp.coo_matrix((data,(rows,cols)), shape=(N,N), dtype=float).tocsr()

    d = np.concatenate([np.ones(nV), -np.ones(nE), np.ones(nF)])
    Ddeg = sp.diags(d, 0, format="csr")
    cdiag = np.concatenate([1j*np.ones(nV), 1j*np.ones(nE), (-1j)*np.ones(nF)])
    C = sp.diags(cdiag, 0, format="csr")
    T = 0.5*(P.getH() + P@Ddeg)
    S = T@C
    return B, P, S, (u, v, edge_to_id, face_edge_ids, inc), (nV, nE, nF)

def build_Hpm_from_BS(B, S, sign=+1):
    return (B + B.getH()).astype(complex) + (sign*S)

def original_basis_indices(nV, nE, nF, nVd, nEd):
    verts_idx = np.arange(nV, dtype=int)
    edge_idx = nVd + np.arange(nE, dtype=int)
    face_idx = nVd + nEd + np.arange(nF, dtype=int)
    return np.concatenate([verts_idx, edge_idx, face_idx])

def boundary_basis_open(faces, nV, nE):
    u, v, edge_to_id, face_edge_ids, inc, _, _ = boundary_data(faces)
    bverts = sorted(boundary_vertices_set(faces))
    bedges = np.where(inc == 1)[0].astype(int)
    return np.concatenate([np.array(bverts, dtype=int), nV + bedges])

def simplex_basis_coords(verts, faces, u, v):
    edge_mid = 0.5*(verts[np.asarray(u, dtype=int)] + verts[np.asarray(v, dtype=int)])
    face_ctr = np.array([np.mean(verts[np.asarray(face, dtype=int)], axis=0) for face in faces])
    return np.vstack([verts, edge_mid, face_ctr])

def simplex_adjacency(nV, nE, nF, u, v, face_edge_ids):
    rows=[]; cols=[]
    for eid,(a,b) in enumerate(zip(u,v)):
        eidx = nV + eid
        for q in (int(a), int(b)):
            rows += [eidx, q]; cols += [q, eidx]
    for fid,eids in enumerate(face_edge_ids):
        fidx = nV + nE + fid
        for eid in eids:
            eidx = nV + int(eid)
            rows += [fidx, eidx]; cols += [eidx, fidx]
    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix((data, (rows, cols)), shape=(nV+nE+nF, nV+nE+nF))

def distances_to_boundary(nV, nE, nF, u, v, face_edge_ids, boundary_basis):
    adj = simplex_adjacency(nV, nE, nF, u, v, face_edge_ids)
    dist = -np.ones(nV+nE+nF, dtype=int)
    frontier = list(map(int, boundary_basis))
    for x in frontier:
        dist[x] = 0
    head = 0
    while head < len(frontier):
        x = frontier[head]; head += 1
        start, end = adj.indptr[x], adj.indptr[x+1]
        for y in adj.indices[start:end]:
            if dist[y] < 0:
                dist[y] = dist[x] + 1
                frontier.append(int(y))
    return dist

def sparse_support_stats(R, dist, tol=1e-10):
    C = R.tocoo()
    mask = np.abs(C.data) > tol
    if not np.any(mask):
        return 0, 0, np.array([], dtype=int), np.array([], dtype=float)
    rows = C.row[mask]
    cols = C.col[mask]
    vals = np.abs(C.data[mask])
    support = np.unique(np.concatenate([rows, cols]))
    pair_dist = np.maximum(dist[rows], dist[cols])
    radius = int(np.max(pair_dist))
    bins = np.arange(radius+1, dtype=int)
    weights = np.array([float(np.sum(vals[pair_dist == k])) for k in bins])
    return len(support), radius, bins, weights

# -------------------------
# Jamming indices
# -------------------------
def jam_indices_for_double(faces_double, uD, vD, edge_to_idD,
                           nVd, nEd, nFd, nF_original, copy_only_vertices):
    inc_faces = edge_incident_faces(faces_double, uD, vD, edge_to_idD)
    copy_only_edge_ids=set()
    shared_edge_ids=set()
    for eid, flist in enumerate(inc_faces):
        if min(flist) >= nF_original:
            copy_only_edge_ids.add(eid)
        if (min(flist) < nF_original) and (max(flist) >= nF_original):
            shared_edge_ids.add(eid)

    jam=[]
    jam += sorted(list(copy_only_vertices))
    jam += [nVd + eid for eid in sorted(list(copy_only_edge_ids))]
    jam += [nVd + nEd + fid for fid in range(nF_original, nFd)]
    jam = np.array(jam, dtype=int)
    return jam, shared_edge_ids

# -------------------------
# Plot helpers
# -------------------------
def plot_mesh_3d(verts, faces, bedges, title, outname):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(6.6, 5.2))
    ax = fig.add_subplot(111, projection='3d')
    x,y,z = verts[:,0], verts[:,1], verts[:,2]
    ax.plot_trisurf(x, y, z, triangles=faces, linewidth=0.2, antialiased=True, alpha=0.85)
    for (a,b,_) in bedges:
        ax.plot([x[a],x[b]],[y[a],y[b]],[z[a],z[b]], linewidth=2.0)
    ax.set_title(title); ax.set_axis_off()
    ax.view_init(elev=22, azim=35)
    return savefig(outname)

def plot_spectrum(evals, g, title, outname):
    plt.figure(figsize=(7.8, 3.6))
    s = np.sort(evals)
    plt.plot(np.arange(len(s)), s, marker='.', linestyle='none', markersize=2.5)
    plt.axhline(+g, linestyle='--', linewidth=1.2)
    plt.axhline(-g, linestyle='--', linewidth=1.2)
    plt.xlabel("sorted index"); plt.ylabel("E"); plt.title(title)
    return savefig(outname)

def plot_jammed_spectrum(evals_j, g, M, title, outname):
    plt.figure(figsize=(7.8, 3.6))
    s=np.sort(evals_j)
    plt.plot(np.arange(len(s)), s, marker='.', linestyle='none', markersize=2.5)
    plt.axhline(+g, linestyle='--', linewidth=1.2)
    plt.axhline(-g, linestyle='--', linewidth=1.2)
    plt.axhline(M, linestyle=':', linewidth=1.0)
    plt.xlabel("sorted index"); plt.ylabel("E"); plt.title(title)
    return savefig(outname)

def plot_boundary_localization(evals, evecs, boundary_basis, g, title, outname):
    weights = np.sum(np.abs(evecs[boundary_basis,:])**2, axis=0)
    plt.figure(figsize=(7.2, 3.8))
    plt.scatter(evals, weights, s=8)
    plt.axvline(+g, linestyle='--', linewidth=1.2)
    plt.axvline(-g, linestyle='--', linewidth=1.2)
    plt.xlabel("E"); plt.ylabel("boundary weight"); plt.title(title)
    return savefig(outname)

def select_edge_subspace(evals, evecs, boundary_basis, g, K=EDGE_K):
    weights = np.sum(np.abs(evecs[boundary_basis,:])**2, axis=0)
    g_eff = g if g > 0 else max(np.percentile(np.abs(evals), 10), 1e-3)

    if g > 0:
        cand = np.where(np.abs(evals) < 2.0*g)[0]
        if cand.size < 6:
            cand = np.arange(len(evals))
    else:
        cand = np.arange(len(evals))

    denom = 1.0 + (evals/(g_eff + 1e-12))**2
    score = weights / denom
    cand_sorted = cand[np.argsort(score[cand])[::-1]]
    sel = cand_sorted[:min(K, cand_sorted.size)]
    return evals[sel], evecs[:, sel]

def boundary_heatmap_and_mean(evals_sub, evecs_sub, boundary_cycles, title, outname,
                              tmax=EDGE_TMAX, nt=EDGE_NT, start_vertex=None):
    if len(boundary_cycles) == 0:
        return save_placeholder_png(outname, title+"\n(no boundary cycles?)"), None, None, None

    cyc_main = max(boundary_cycles, key=len)
    cyc_main = np.array(cyc_main, dtype=int)
    L = len(cyc_main)
    if start_vertex is None:
        start_vertex = int(cyc_main[0])

    N = evecs_sub.shape[0]
    e0 = np.zeros(N, dtype=complex); e0[start_vertex]=1.0
    c0 = evecs_sub.conj().T @ e0
    c0 = c0 / (np.linalg.norm(c0)+1e-15)

    ts = np.linspace(0.0, tmax, nt)
    heatmaps=[]
    for cyc in boundary_cycles:
        cyc=np.array(cyc, dtype=int)
        Hm=np.zeros((nt, len(cyc)))
        for it,t in enumerate(ts):
            psi_t = evecs_sub @ (np.exp(-1j*evals_sub*t) * c0)
            Hm[it,:] = np.abs(psi_t[cyc])**2
        heatmaps.append(Hm)

    angles = np.exp(2j*np.pi*np.arange(L)/L)
    means=[]
    for t in ts:
        psi_t = evecs_sub @ (np.exp(-1j*evals_sub*t) * c0)
        p = np.abs(psi_t[cyc_main])**2
        if p.sum()<1e-14:
            means.append(0.0)
        else:
            z = np.sum(p*angles)/np.sum(p)
            means.append((np.angle(z)/(2*np.pi)*L) % L)
    means=np.array(means)

    un=means.copy()
    for i in range(1,len(un)):
        while un[i]-un[i-1] > 0.5*L: un[i]-=L
        while un[i]-un[i-1] < -0.5*L: un[i]+=L

    fig = plt.figure(figsize=(7.8, 2.2+2.0*len(boundary_cycles)))
    for k,Hm in enumerate(heatmaps):
        ax = fig.add_subplot(len(boundary_cycles),1,k+1)
        ax.imshow(Hm, aspect='auto', origin='lower')
        ax.set_ylabel("t index"); ax.set_xlabel("bdry idx")
        ax.set_title(f"{title} (boundary {k+1})")
    heat_path = savefig(outname)

    return heat_path, ts, un, L

def fit_drift_speed(ts, unwrapped_mean):
    if ts is None or unwrapped_mean is None or len(ts)<3:
        return None
    slope, _ = np.polyfit(ts, unwrapped_mean, 1)
    return float(slope)

# -------------------------
# Resolvent convergence + leakage + eig-localization (ex1)
# -------------------------
def resolvent_convergence_ex1(Hplus_dense, orig_idx, copy_idx, Ms, z=1j):
    A = Hplus_dense[np.ix_(orig_idx, orig_idx)]
    C = Hplus_dense[np.ix_(orig_idx, copy_idx)]
    D = Hplus_dense[np.ix_(copy_idx, copy_idx)]
    Iorig = np.eye(A.shape[0], dtype=complex)
    Icopy = np.eye(D.shape[0], dtype=complex)
    R0 = la.solve(A - z*Iorig, Iorig, assume_a='gen')

    deltas=[]; invM=[]
    for M in Ms:
        Rcopy = la.solve(D + (M - z)*Icopy, Icopy, assume_a='gen')
        K = C @ Rcopy @ C.conj().T
        Rblock = la.solve(A - z*Iorig - K, Iorig, assume_a='gen')
        deltas.append(float(la.norm(Rblock - R0, 2)))
        invM.append(1.0/float(M))

    plt.figure(figsize=(7.2,3.8))
    plt.plot(invM, deltas, "-o")
    plt.xlabel("1/M"); plt.ylabel("resolvent diff (spectral norm)")
    plt.title("Resolvent convergence (z=i), ex1")
    return savefig("fig_resolvent_convergence.png"), np.array(invM), np.array(deltas)

def leakage_vs_M_ex1(Hplus_dense, jam_idx, copy_idx, g, Ms):
    if g <= 0:
        return save_placeholder_png("fig_ex1_leakage_vs_M.png", "Leakage vs M skipped (g≈0)."), None, None, None

    leaks=[]; invM=[]; leak_bounds=[]
    N = Hplus_dense.shape[0]
    orig_idx = np.array(sorted(list(set(range(N)) - set(jam_idx))), dtype=int)
    C = Hplus_dense[np.ix_(orig_idx, copy_idx)]
    D = Hplus_dense[np.ix_(copy_idx, copy_idx)]

    C_norm = float(la.svdvals(C)[0]) if C.size else 0.0
    D_norm = float(np.max(np.abs(la.eigvalsh(D)))) if D.size else 0.0

    for M in Ms:
        Hjam = Hplus_dense.copy()
        Hjam[jam_idx, jam_idx] += M
        evals, evecs = la.eigh(Hjam)

        ingap = np.where(np.abs(evals) < g)[0]
        if ingap.size == 0:
            leaks.append(np.nan); leak_bounds.append(np.nan); invM.append(1.0/M); continue

        copy_w = np.sum(np.abs(evecs[copy_idx,:])**2, axis=0)
        leaks.append(float(np.mean(copy_w[ingap])))
        invM.append(1.0/M)

        denom = (M - D_norm - g)
        leak_bounds.append(float((C_norm/denom)**2) if (denom>0 and C_norm>0) else np.nan)

    plt.figure(figsize=(7.2,3.8))
    plt.plot(invM, leaks, "-o", label="mean copy weight (in-gap)")
    plt.plot(invM, leak_bounds, "--", label="(||C||/(M-||D||-g))^2")
    plt.xlabel("1/M"); plt.ylabel("mean copy weight")
    plt.title("Leakage scaling vs 1/M (ex1)")
    plt.legend()
    return savefig("fig_ex1_leakage_vs_M.png"), np.array(invM), np.array(leaks), np.array(leak_bounds)

def _nearest_dist_to_sorted_spectrum(E, spec_sorted):
    j = np.searchsorted(spec_sorted, E)
    if j <= 0: return abs(E - spec_sorted[0])
    if j >= len(spec_sorted): return abs(E - spec_sorted[-1])
    return min(abs(E - spec_sorted[j-1]), abs(E - spec_sorted[j]))

def eig_localization_vs_M_ex1(Hplus_dense, orig_idx, copy_idx, jam_idx, g, Ms):
    fig_name = "fig_ex1_eig_localization_vs_M.png"
    tex_name = "eig_localization_summary.tex"
    tex_path = os.path.join(OUTDIR, tex_name)

    if g <= 0:
        fig_path = save_placeholder_png(fig_name, "Eigenvalue localization skipped (g≈0).")
        with open(tex_path, "w") as f:
            f.write(r"\begin{tabular}{@{}l@{}}\toprule Skipped ($g\approx 0$).\\\bottomrule\end{tabular}"+"\n")
        return fig_path, tex_path

    A = Hplus_dense[np.ix_(orig_idx, orig_idx)]
    C = Hplus_dense[np.ix_(orig_idx, copy_idx)]
    D = Hplus_dense[np.ix_(copy_idx, copy_idx)]
    evalsA = np.sort(la.eigvalsh(A))

    C_norm = float(la.svdvals(C)[0]) if C.size else 0.0
    D_norm = float(np.max(np.abs(la.eigvalsh(D)))) if D.size else 0.0
    E_max = g

    xs=[]; ys=[]; yb=[]; rows=[]
    for M in Ms:
        Hjam = Hplus_dense.copy()
        Hjam[jam_idx, jam_idx] += M
        evals = la.eigvalsh(Hjam)

        ingap = evals[np.abs(evals) < g]
        n_in = int(ingap.size)
        max_dev = np.nan if n_in==0 else float(np.max([_nearest_dist_to_sorted_spectrum(float(E), evalsA) for E in ingap]))

        denom = M - D_norm - E_max
        bound = (C_norm**2/denom) if (denom>0 and C_norm>0) else np.nan

        xs.append(1.0/M); ys.append(max_dev); yb.append(bound)
        rows.append((M, 1.0/M, n_in, max_dev, bound))

    plt.figure(figsize=(7.2,3.8))
    plt.plot(xs, ys, "-o", label="measured max dist to spec(A)")
    plt.plot(xs, yb, "--", label="bound: ||C||^2/(M-||D||-g)")
    plt.xlabel("1/M"); plt.ylabel("max deviation")
    plt.title("Example 1: eigenvalue localization vs 1/M")
    plt.legend()
    fig_path = savefig(fig_name)

    with open(tex_path, "w") as f:
        f.write(r"\begin{tabular}{@{}rrrrr@{}}"+"\n")
        f.write(r"\toprule"+"\n")
        f.write(r"$M$ & $1/M$ & \# in-gap & $\max_{|E|<g}\mathrm{dist}(E,\mathrm{spec}(A))$ & bound \\"+"\n")
        f.write(r"\midrule"+"\n")
        for (M, invM, n_in, max_dev, bound) in rows:
            f.write(f"{M:.1f} & {invM:.4f} & {n_in} & ${_tex_num(max_dev)}$ & ${_tex_num(bound)}$ \\\\\n")
        f.write(r"\bottomrule"+"\n")
        f.write(r"\end{tabular}"+"\n")

    return fig_path, tex_path

# -------------------------
# Refinement scaling (sparse)
# -------------------------
def estimate_gap_sparse(H, k=18):
    try:
        vals = spla.eigsh(H, k=k, sigma=0.0, which="LM", return_eigenvectors=False)
        return robust_half_gap(vals)[1]
    except Exception:
        try:
            vals = spla.eigsh(H, k=k, which="SM", return_eigenvectors=False)
            return robust_half_gap(vals)[1]
        except Exception:
            return np.nan

def refinement_scaling_figure(levels=(1,2,3)):
    gaps = {"ex1":[], "ex2":[], "ex3":[]}
    lv   = {"ex1":[], "ex2":[], "ex3":[]}

    for s in levels:
        v0, f0 = icosahedron()
        vS, fS = subdivide_icosphere(v0, f0, s)

        try:
            v1_, f1_, *_ = build_ex1_cap(vS, fS, z_cut=Z_CUT_CAP)
            vD1_, fD1_, *_ = build_double(v1_, f1_)
            B1_,P1_,S1_,_,_ = build_B_P_S(vD1_, fD1_)
            H1_ = build_Hpm_from_BS(B1_,S1_,sign=+1)
            g1_ = estimate_gap_sparse(H1_, k=22)
            if np.isfinite(g1_): gaps["ex1"].append(g1_); lv["ex1"].append(s)
        except Exception as e:
            warnings.warn(f"[scaling] skip ex1 at s={s}: {e}")

        try:
            v2_, f2_, *_ = build_ex2_band(vS, fS, z_band=Z_BAND)
            vD2_, fD2_, *_ = build_double(v2_, f2_)
            B2_,P2_,S2_,_,_ = build_B_P_S(vD2_, fD2_)
            H2_ = build_Hpm_from_BS(B2_,S2_,sign=+1)
            g2_ = estimate_gap_sparse(H2_, k=22)
            if np.isfinite(g2_): gaps["ex2"].append(g2_); lv["ex2"].append(s)
        except Exception as e:
            warnings.warn(f"[scaling] skip ex2 at s={s}: {e}")

        try:
            v3_, f3_, *_ = build_ex3_two_holes(vS, fS, z_cut=Z_CUT_CAP, centers=(C1,C2), r_hint=R_HINT_EX3)
            vD3_, fD3_, *_ = build_double(v3_, f3_)
            B3_,P3_,S3_,_,_ = build_B_P_S(vD3_, fD3_)
            H3_ = build_Hpm_from_BS(B3_,S3_,sign=+1)
            g3_ = estimate_gap_sparse(H3_, k=24)
            if np.isfinite(g3_): gaps["ex3"].append(g3_); lv["ex3"].append(s)
        except Exception as e:
            warnings.warn(f"[scaling] skip ex3 at s={s}: {e}")

    plt.figure(figsize=(7.2,3.8))
    if lv["ex1"]: plt.plot(lv["ex1"], gaps["ex1"], "-o", label="ex1")
    if lv["ex2"]: plt.plot(lv["ex2"], gaps["ex2"], "-o", label="ex2")
    if lv["ex3"]: plt.plot(lv["ex3"], gaps["ex3"], "-o", label="ex3")
    plt.xlabel("subdivision level"); plt.ylabel("half-gap g"); plt.title("Refinement scaling")
    plt.legend()
    return savefig("fig_refinement_scaling.png")

# -------------------------
# MAIN RUN (paper default)
# -------------------------
all_files = []

verts0, faces0 = icosahedron()
vertsS, facesS = subdivide_icosphere(verts0, faces0, SUBDIV_DEFAULT)

# ---- Example 1
v1, f1, cycles1, bV1, bE1, rounds1 = build_ex1_cap(vertsS, facesS, z_cut=Z_CUT_CAP)
bedges1 = boundary_data(f1)[6]
all_files.append(plot_mesh_3d(v1, f1, bedges1, f"Example 1 mesh (cap, 1 boundary) [bary={rounds1}]",
                              "fig_ex1_mesh.png"))

vD1, fD1, cycles1D, boundary_vertices1, copy_map1, copy_only_vertices1 = build_double(v1, f1)
B1, P1, S1, (uD1, vD1e, edge_to_idD1), (nVd1, nEd1, nFd1) = build_B_P_S(vD1, fD1)
Hplus1 = build_Hpm_from_BS(B1,S1,sign=+1).toarray()
Hminus1= build_Hpm_from_BS(B1,S1,sign=-1).toarray()

evals_double1 = la.eigvalsh(Hplus1)
g_raw1, g1 = robust_half_gap(evals_double1)

nF_orig1 = f1.shape[0]
jam_idx1, shared_edge_ids1 = jam_indices_for_double(
    fD1, uD1, vD1e, edge_to_idD1, nVd1, nEd1, nFd1, nF_orig1, copy_only_vertices1
)
copy_idx1 = np.array(sorted(list(jam_idx1)), dtype=int)
orig_idx1 = np.array(sorted(list(set(range(Hplus1.shape[0])) - set(jam_idx1))), dtype=int)

Hjam1 = Hplus1.copy()
Hjam1[jam_idx1, jam_idx1] += JAM_M_DEFAULT
evals_j1, evecs_j1 = la.eigh(Hjam1)
in_gap1 = int(np.sum(np.abs(evals_j1) < g1)) if g1 > 0 else 0

all_files.append(plot_spectrum(evals_double1, g1, "Example 1: spectrum of H_+(D(K))", "fig_ex1_bulk_spectrum.png"))
all_files.append(plot_jammed_spectrum(evals_j1, g1, JAM_M_DEFAULT, "Example 1: spectrum of H_{+,M}", "fig_ex1_jammed_spectrum.png"))

boundary_vertex_basis1 = np.array(sorted(list(boundary_vertices1)), dtype=int)
boundary_edge_basis1 = np.array([nVd1 + eid for eid in sorted(list(shared_edge_ids1))], dtype=int)
boundary_basis1 = np.concatenate([boundary_vertex_basis1, boundary_edge_basis1])
all_files.append(plot_boundary_localization(evals_j1, evecs_j1, boundary_basis1, g1,
                                           "Example 1: boundary localization", "fig_ex1_boundary_localization.png"))

evals_edge1, evecs_edge1 = select_edge_subspace(evals_j1, evecs_j1, boundary_basis1, g1, K=EDGE_K)
heat1, ts1, mean1, L1 = boundary_heatmap_and_mean(evals_edge1, evecs_edge1, cycles1,
                                                  "Example 1 (H_{+,M})", "fig_ex1_edge_propagation.png")
all_files.append(heat1)
vplus = fit_drift_speed(ts1, mean1)

Hjam1m = Hminus1.copy()
Hjam1m[jam_idx1, jam_idx1] += JAM_M_DEFAULT
evals_j1m, evecs_j1m = la.eigh(Hjam1m)
evals_edge1m, evecs_edge1m = select_edge_subspace(evals_j1m, evecs_j1m, boundary_basis1, g1, K=EDGE_K)
_, ts1m, mean1m, _ = boundary_heatmap_and_mean(evals_edge1m, evecs_edge1m, cycles1, "tmp", "tmp.png")
try:
    tmp_path = os.path.join(OUTDIR, "tmp.png")
    if os.path.exists(tmp_path): os.remove(tmp_path)
except Exception:
    pass
vminus = fit_drift_speed(ts1m, mean1m)

plt.figure(figsize=(7.2,3.6))
if ts1 is not None and mean1 is not None: plt.plot(ts1, mean1, linewidth=2.0, label="H_{+,M}")
if ts1m is not None and mean1m is not None: plt.plot(ts1m, mean1m, linewidth=2.0, label="H_{-,M}")
plt.xlabel("t"); plt.ylabel("mean boundary index (unwrapped)")
plt.title("Example 1: chirality flip (mean position)")
plt.legend()
all_files.append(savefig("fig_ex1_chirality_flip.png"))

print(f"ex1: bary_rounds={rounds1} boundaries={len(cycles1)} sizes={sorted([len(c) for c in cycles1], reverse=True)} "
      f"g_raw={g_raw1:.4e} g={g1:.4e} in-gap={in_gap1} v+={vplus:.3f} v-={vminus:.3f}")

# ---- Example 2
v2, f2, cycles2, bV2, bE2, rounds2 = build_ex2_band(vertsS, facesS, z_band=Z_BAND)
bedges2 = boundary_data(f2)[6]
all_files.append(plot_mesh_3d(v2, f2, bedges2, f"Example 2 mesh (band, 2 boundaries) [bary={rounds2}]",
                              "fig_ex2_mesh.png"))

vD2, fD2, cycles2D, boundary_vertices2, copy_map2, copy_only_vertices2 = build_double(v2, f2)
B2, P2, S2, (uD2, vD2e, edge_to_idD2), (nVd2, nEd2, nFd2) = build_B_P_S(vD2, fD2)
Hplus2 = build_Hpm_from_BS(B2,S2,sign=+1).toarray()
evals_double2 = la.eigvalsh(Hplus2)
g_raw2, g2 = robust_half_gap(evals_double2)

nF_orig2 = f2.shape[0]
jam_idx2, shared_edge_ids2 = jam_indices_for_double(
    fD2, uD2, vD2e, edge_to_idD2, nVd2, nEd2, nFd2, nF_orig2, copy_only_vertices2
)
Hjam2 = Hplus2.copy()
Hjam2[jam_idx2, jam_idx2] += JAM_M_DEFAULT
evals_j2, evecs_j2 = la.eigh(Hjam2)
in_gap2 = int(np.sum(np.abs(evals_j2) < g2)) if g2 > 0 else 0

all_files.append(plot_spectrum(evals_double2, g2, "Example 2: spectrum of H_+(D(K))", "fig_ex2_bulk_spectrum.png"))
all_files.append(plot_jammed_spectrum(evals_j2, g2, JAM_M_DEFAULT, "Example 2: spectrum of H_{+,M}", "fig_ex2_jammed_spectrum.png"))

boundary_vertex_basis2 = np.array(sorted(list(boundary_vertices2)), dtype=int)
boundary_edge_basis2 = np.array([nVd2 + eid for eid in sorted(list(shared_edge_ids2))], dtype=int)
boundary_basis2 = np.concatenate([boundary_vertex_basis2, boundary_edge_basis2])
evals_edge2, evecs_edge2 = select_edge_subspace(evals_j2, evecs_j2, boundary_basis2, g2, K=EDGE_K)
heat2, *_ = boundary_heatmap_and_mean(evals_edge2, evecs_edge2, cycles2,
                                      "Example 2 (H_{+,M})", "fig_ex2_edge_propagation.png")
all_files.append(heat2)

print(f"ex2: bary_rounds={rounds2} boundaries={len(cycles2)} sizes={sorted([len(c) for c in cycles2], reverse=True)} "
      f"g_raw={g_raw2:.4e} g={g2:.4e} in-gap={in_gap2}")

# ---- Example 3
v3, f3, cycles3, bV3, bE3, r_used3, rounds3 = build_ex3_two_holes(
    vertsS, facesS, z_cut=Z_CUT_CAP, centers=(C1,C2), r_hint=R_HINT_EX3
)
bedges3 = boundary_data(f3)[6]
all_files.append(plot_mesh_3d(v3, f3, bedges3, f"Example 3 mesh (cap+2 holes, r={r_used3:.3f}) [bary={rounds3}]",
                              "fig_ex3_mesh.png"))

vD3, fD3, cycles3D, boundary_vertices3, copy_map3, copy_only_vertices3 = build_double(v3, f3)
B3, P3, S3, (uD3, vD3e, edge_to_idD3), (nVd3, nEd3, nFd3) = build_B_P_S(vD3, fD3)
Hplus3 = build_Hpm_from_BS(B3,S3,sign=+1).toarray()
evals_double3 = la.eigvalsh(Hplus3)
g_raw3, g3 = robust_half_gap(evals_double3)

nF_orig3 = f3.shape[0]
jam_idx3, shared_edge_ids3 = jam_indices_for_double(
    fD3, uD3, vD3e, edge_to_idD3, nVd3, nEd3, nFd3, nF_orig3, copy_only_vertices3
)
Hjam3 = Hplus3.copy()
Hjam3[jam_idx3, jam_idx3] += JAM_M_DEFAULT
evals_j3, evecs_j3 = la.eigh(Hjam3)
in_gap3 = int(np.sum(np.abs(evals_j3) < g3)) if g3 > 0 else 0

all_files.append(plot_spectrum(evals_double3, g3, "Example 3: spectrum of H_+(D(K))", "fig_ex3_bulk_spectrum.png"))
all_files.append(plot_jammed_spectrum(evals_j3, g3, JAM_M_DEFAULT, "Example 3: spectrum of H_{+,M}", "fig_ex3_jammed_spectrum.png"))

boundary_vertex_basis3 = np.array(sorted(list(boundary_vertices3)), dtype=int)
boundary_edge_basis3 = np.array([nVd3 + eid for eid in sorted(list(shared_edge_ids3))], dtype=int)
boundary_basis3 = np.concatenate([boundary_vertex_basis3, boundary_edge_basis3])
evals_edge3, evecs_edge3 = select_edge_subspace(evals_j3, evecs_j3, boundary_basis3, g3, K=EDGE_K)
heat3, *_ = boundary_heatmap_and_mean(evals_edge3, evecs_edge3, cycles3,
                                      "Example 3 (H_{+,M})", "fig_ex3_edge_propagation.png")
all_files.append(heat3)

print(f"ex3: bary_rounds={rounds3} boundaries={len(cycles3)} sizes={sorted([len(c) for c in cycles3], reverse=True)} "
      f"g_raw={g_raw3:.4e} g={g3:.4e} in-gap={in_gap3}")

# -------------------------
# Tables (.tex)
# -------------------------
def mesh_counts(verts, faces):
    u, v, edge_to_id, face_edge_ids = compute_edges_from_faces(faces)
    return verts.shape[0], len(u), faces.shape[0]

def write_mesh_stats_table():
    rows = []
    for tag, v_patch, f_patch, cycles, bV, bE, vD, fD, gval in [
        ("ex1", v1, f1, cycles1, bV1, bE1, vD1, fD1, g1),
        ("ex2", v2, f2, cycles2, bV2, bE2, vD2, fD2, g2),
        ("ex3", v3, f3, cycles3, bV3, bE3, vD3, fD3, g3),
    ]:
        nV,nE,nF = mesh_counts(v_patch, f_patch)
        nVd,nEd,nFd = mesh_counts(vD, fD)
        dimH  = nV+nE+nF
        dimHd = nVd+nEd+nFd
        rows.append((tag, len(cycles), bV, bE, nV,nE,nF, dimH, nVd,nEd,nFd, dimHd, gval))

    table_path = os.path.join(OUTDIR,"mesh_stats_table.tex")
    with open(table_path,"w") as f:
        f.write(r"\begin{tabular}{@{}lrrrrrrrrrrrr@{}}"+"\n")
        f.write(r"\toprule"+"\n")
        f.write(r"ex & $\#\partial$ & $|\partial\K_0|$ & $|\partial\K_1|$ & $|\K_0|$ & $|\K_1|$ & $|\K_2|$ & $\dim\Hil(\K)$ & $|D_0|$ & $|D_1|$ & $|D_2|$ & $\dim\Hil(D)$ & $g$ \\"+"\n")
        f.write(r"\midrule"+"\n")
        for (tag, nb, bV, bE, nV,nE,nF, dimH, nVd,nEd,nFd, dimHd, gval) in rows:
            f.write(
                f"{tag} & {nb} & {bV} & {bE} & {nV} & {nE} & {nF} & {dimH} & "
                f"{nVd} & {nEd} & {nFd} & {dimHd} & ${_tex_num(gval)}$ \\\\\n"
            )
        f.write(r"\bottomrule"+"\n")
        f.write(r"\end{tabular}"+"\n")
    return table_path

def fmt_float_or_dash(x, nd=3):
    if x is None: return r"--"
    try:
        x = float(x)
    except Exception:
        return r"--"
    return r"--" if (not np.isfinite(x)) else f"{x:.{nd}f}"

def write_numerics_summary():
    table_path = os.path.join(OUTDIR,"numerics_summary.tex")
    with open(table_path,"w") as f:
        f.write(r"\begin{tabular}{@{}lrrrr@{}}"+"\n")
        f.write(r"\toprule"+"\n")
        f.write(r"ex & $g$ & \# in-gap ($|E|<g$) & $v_{+}$ & $v_{-}$ \\"+"\n")
        f.write(r"\midrule"+"\n")
        f.write(f"ex1 & ${_tex_num(g1)}$ & {in_gap1} & {fmt_float_or_dash(vplus)} & {fmt_float_or_dash(vminus)} \\\\\n")
        f.write(f"ex2 & ${_tex_num(g2)}$ & {in_gap2} & -- & -- \\\\\n")
        f.write(f"ex3 & ${_tex_num(g3)}$ & {in_gap3} & -- & -- \\\\\n")
        f.write(r"\bottomrule"+"\n")
        f.write(r"\end{tabular}"+"\n")
    return table_path

mesh_table_path = write_mesh_stats_table(); all_files.append(mesh_table_path)
num_table_path  = write_numerics_summary();  all_files.append(num_table_path)

# -------------------------
# Fig 7,8,11 equivalents (ex1)
# -------------------------
fig_res, *_ = resolvent_convergence_ex1(Hplus1, orig_idx1, copy_idx1, M_LIST_RESOLVENT, z=1j)
all_files.append(fig_res)

fig_leak, *_ = leakage_vs_M_ex1(Hplus1, jam_idx1, copy_idx1, g1, M_LIST_LEAKAGE)
all_files.append(fig_leak)

fig_eigloc, tex_eigloc = eig_localization_vs_M_ex1(Hplus1, orig_idx1, copy_idx1, jam_idx1, g1, M_LIST_EIGLOC)
all_files.append(fig_eigloc); all_files.append(tex_eigloc)

# -------------------------
# Fig 9 refinement trend
# -------------------------
all_files.append(refinement_scaling_figure(levels=(1,2,3)))

# -------------------------
# Fig 10 disorder test (ex1)
# -------------------------
if g1 > 0:
    eps = DISORDER_FRAC * g1
    Hdis = Hjam1.copy()
    Hdis[np.arange(Hdis.shape[0]), np.arange(Hdis.shape[0])] += np.random.uniform(-eps, eps, size=Hdis.shape[0])
    evals_dis, evecs_dis = la.eigh(Hdis)
    evals_edge_dis, evecs_edge_dis = select_edge_subspace(evals_dis, evecs_dis, boundary_basis1, g1, K=EDGE_K)
    heat_dis, *_ = boundary_heatmap_and_mean(evals_edge_dis, evecs_edge_dis, cycles1,
                                             f"Example 1 disorder (eps={eps:.3f})",
                                             "fig_ex1_edge_propagation_disorder.png")
    all_files.append(heat_dis)
else:
    all_files.append(save_placeholder_png("fig_ex1_edge_propagation_disorder.png",
                                          "Example 1 disorder skipped (g≈0)."))

# -------------------------
# Additional high-impact robustness check: nonconstant positive copy-side confinement
# -------------------------
def copy_potential_robustness_ex1():
    rng = np.random.default_rng(20260516)
    Hrand = Hplus1.copy()
    xi = rng.uniform(-0.35, 0.35, size=len(copy_idx1))
    random_potential = JAM_M_DEFAULT * (1.0 + xi)
    Hrand[copy_idx1, copy_idx1] += random_potential
    evals_rand, evecs_rand = la.eigh(Hrand)

    scalar_ingap = np.where(np.abs(evals_j1) < g1)[0] if g1 > 0 else np.array([], dtype=int)
    random_ingap = np.where(np.abs(evals_rand) < g1)[0] if g1 > 0 else np.array([], dtype=int)

    scalar_bw = np.sum(np.abs(evecs_j1[boundary_basis1,:])**2, axis=0)
    random_bw = np.sum(np.abs(evecs_rand[boundary_basis1,:])**2, axis=0)
    scalar_cw = np.sum(np.abs(evecs_j1[copy_idx1,:])**2, axis=0)
    random_cw = np.sum(np.abs(evecs_rand[copy_idx1,:])**2, axis=0)

    svals = np.sort(evals_j1[np.abs(evals_j1) < 1.35*g1]) if g1 > 0 else np.array([])
    rvals = np.sort(evals_rand[np.abs(evals_rand) < 1.35*g1]) if g1 > 0 else np.array([])

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax[0].axhline(g1, color='k', linestyle='--', linewidth=1)
    ax[0].axhline(-g1, color='k', linestyle='--', linewidth=1)
    if svals.size:
        ax[0].scatter(np.zeros_like(svals), svals, label='scalar M', s=26)
    if rvals.size:
        ax[0].scatter(np.ones_like(rvals), rvals, label='nonconstant positive V', s=26)
    ax[0].set_xticks([0,1], ['scalar', 'nonconstant'])
    ax[0].set_ylabel('E')
    ax[0].set_title('Low-energy spectrum')
    ax[0].legend(fontsize=8)

    labels = ['scalar', 'nonconstant']
    mean_boundary = [float(np.mean(scalar_bw[scalar_ingap])) if scalar_ingap.size else np.nan,
                     float(np.mean(random_bw[random_ingap])) if random_ingap.size else np.nan]
    mean_copy = [float(np.mean(scalar_cw[scalar_ingap])) if scalar_ingap.size else np.nan,
                 float(np.mean(random_cw[random_ingap])) if random_ingap.size else np.nan]
    x = np.arange(2)
    width = 0.35
    ax[1].bar(x-width/2, mean_boundary, width, label='boundary weight')
    ax[1].bar(x+width/2, mean_copy, width, label='copy weight')
    ax[1].set_xticks(x, labels)
    ax[1].set_ylim(0, max(0.08, np.nanmax(mean_boundary + mean_copy)*1.25))
    ax[1].set_title('Mean in-gap weights')
    ax[1].legend(fontsize=8)
    fig.suptitle('Example 1: robustness under nonconstant positive copy-side confinement')
    fig.tight_layout()
    fig_path = savefig('fig_copy_confinement_robustness.png')

    tex_path = os.path.join(OUTDIR, 'copy_confinement_robustness_table.tex')
    with open(tex_path, 'w') as f:
        f.write(r'\begin{tabular}{@{}lrrrr@{}}'+'\n')
        f.write(r'\toprule'+'\n')
        f.write(r'copy potential & \# in-gap & mean boundary weight & mean copy weight & potential range \\'+'\n')
        f.write(r'\midrule'+'\n')
        f.write(f"scalar $M$ & {scalar_ingap.size} & ${_tex_num(mean_boundary[0])}$ & ${_tex_num(mean_copy[0])}$ & $[{JAM_M_DEFAULT:.1f},{JAM_M_DEFAULT:.1f}]$ \\\\\n")
        f.write(f"nonconstant $V$ & {random_ingap.size} & ${_tex_num(mean_boundary[1])}$ & ${_tex_num(mean_copy[1])}$ & $[{float(np.min(random_potential)):.1f},{float(np.max(random_potential)):.1f}]$ \\\\\n")
        f.write(r'\bottomrule'+'\n')
        f.write(r'\end{tabular}'+'\n')
    print('copy confinement robustness:', fig_path, tex_path)
    return fig_path, tex_path

def pl_boundary_comparison_ex1():
    Brel, Prel, Srel, (u1, v1e, edge_to_id1, face_edge_ids1, inc1), (nV1, nE1, nF1) = build_B_P_S_relative(v1, f1)
    Hpl = build_Hpm_from_BS(Brel, Srel, sign=+1).tocsr()
    basis_orig = original_basis_indices(nV1, nE1, nF1, nVd1, nEd1)
    Hpartial = sp.csr_matrix(Hplus1[np.ix_(basis_orig, basis_orig)])
    Hjam_low = Hjam1
    R = (Hpartial - Hpl).tocsr()

    bd_basis = boundary_basis_open(f1, nV1, nE1)
    dist = distances_to_boundary(nV1, nE1, nF1, u1, v1e, face_edge_ids1, bd_basis)
    support_dim, support_radius, bins, weights = sparse_support_stats(R, dist)
    dimH = nV1+nE1+nF1
    svals = la.svdvals(R.toarray())
    rank_R = int(np.sum(svals > 1e-9))
    op_R = float(svals[0]) if svals.size else 0.0
    fro_R = float(la.norm(R.toarray(), 'fro'))

    evals_pl = np.sort(la.eigvalsh(Hpl.toarray()))
    evals_partial = np.sort(la.eigvalsh(Hpartial.toarray()))
    evals_jam = np.sort(la.eigvalsh(Hjam_low))
    k = 28
    def central(e):
        j = np.argsort(np.abs(e))[:k]
        return np.sort(e[j])
    c_pl = central(evals_pl)
    c_partial = central(evals_partial)
    c_jam = central(evals_jam)

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9))
    ax[0].plot(c_pl, 'o-', markersize=3, label=r'$H^{PL}_+$')
    ax[0].plot(c_partial, 's-', markersize=3, label=r'$H^\partial_+$')
    ax[0].plot(c_jam, '^-', markersize=3, label=r'$H_{+,M}$')
    ax[0].axhline(g1, color='k', linestyle='--', linewidth=0.9)
    ax[0].axhline(-g1, color='k', linestyle='--', linewidth=0.9)
    ax[0].set_xlabel('central eigenvalue index')
    ax[0].set_ylabel('E')
    ax[0].set_title('Central spectra')
    ax[0].legend(fontsize=8)

    if bins.size:
        ax[1].bar(bins, weights)
    ax[1].set_xlabel('max distance to boundary')
    ax[1].set_ylabel(r'$\sum |R_{ij}|$')
    ax[1].set_title(r'Collar support of $R_{\partial,+}$')
    fig.suptitle('Example 1: intrinsic PL model, doubled compression, and jamming')
    fig.tight_layout()
    fig_path = savefig('fig_pl_boundary_comparison.png')

    tex_path = os.path.join(OUTDIR, 'pl_boundary_comparison_table.tex')
    with open(tex_path, 'w') as f:
        f.write(r'\begin{tabular}{@{}lrrrrrr@{}}'+'\n')
        f.write(r'\toprule'+'\n')
        f.write(r'example & $\dim\Hil(\K)$ & $\operatorname{rank}R$ & support dim. & support radius & $\|R\|$ & $\|R\|_F$ \\'+'\n')
        f.write(r'\midrule'+'\n')
        f.write(f"ex1 & {dimH} & {rank_R} & {support_dim} & {support_radius} & ${_tex_num(op_R)}$ & ${_tex_num(fro_R)}$ \\\\\n")
        f.write(r'\bottomrule'+'\n')
        f.write(r'\end{tabular}'+'\n')
    print('PL boundary comparison:', fig_path, tex_path,
          f'rank={rank_R} support_dim={support_dim} radius={support_radius} ||R||={op_R:.4g}')
    return fig_path, tex_path

def twisted_sparse_hamiltonian(H_sparse, coords, theta):
    H = H_sparse.tocoo()
    angles = np.arctan2(coords[:,1], coords[:,0])
    raw = angles[H.row] - angles[H.col]
    sign = np.zeros_like(raw, dtype=float)
    sign[raw > np.pi] = -1.0
    sign[raw < -np.pi] = +1.0
    data = H.data * np.exp(1j * theta * sign)
    return sp.coo_matrix((data, (H.row, H.col)), shape=H_sparse.shape).tocsr()

def flux_spectral_flow_ex2():
    Hbase = build_Hpm_from_BS(B2, S2, sign=+1).tocsr()
    Hbase = Hbase + sp.diags(np.isin(np.arange(Hbase.shape[0]), jam_idx2).astype(float) * JAM_M_DEFAULT, format='csr')
    coords = simplex_basis_coords(vD2, fD2, uD2, vD2e)
    thetas = np.linspace(0.0, 2*np.pi, FLUX_N_THETA)
    vals_by_theta = []
    crossings = 0
    for theta in thetas:
        Htw = twisted_sparse_hamiltonian(Hbase, coords, theta)
        try:
            vals = spla.eigsh(Htw, k=FLUX_K, sigma=0.0, which='LM', return_eigenvectors=False)
        except Exception:
            vals = la.eigvalsh(Htw.toarray())
            vals = vals[np.argsort(np.abs(vals))[:FLUX_K]]
        vals_by_theta.append(np.sort(np.real(vals)))
    vals_by_theta = np.array(vals_by_theta)
    for j in range(vals_by_theta.shape[1]):
        y = vals_by_theta[:, j]
        crossings += int(np.sum((y[:-1] < 0) & (y[1:] >= 0)) + np.sum((y[:-1] > 0) & (y[1:] <= 0)))

    plt.figure(figsize=(8.0, 4.2))
    for j in range(vals_by_theta.shape[1]):
        plt.plot(thetas/(2*np.pi), vals_by_theta[:,j], color='tab:blue', alpha=0.55, linewidth=1.0)
    plt.axhline(g2, color='k', linestyle='--', linewidth=0.9)
    plt.axhline(-g2, color='k', linestyle='--', linewidth=0.9)
    plt.axhline(0, color='k', linewidth=0.7)
    plt.xlabel(r'flux $\theta/2\pi$')
    plt.ylabel('near-zero eigenvalues')
    plt.title('Example 2: finite-size flux-insertion spectral flow')
    fig_path = savefig('fig_flux_spectral_flow_ex2.png')

    min_abs = float(np.min(np.abs(vals_by_theta)))
    max_span = float(np.max(vals_by_theta) - np.min(vals_by_theta))
    tex_path = os.path.join(OUTDIR, 'flux_spectral_flow_table.tex')
    with open(tex_path, 'w') as f:
        f.write(r'\begin{tabular}{@{}lrrrr@{}}'+'\n')
        f.write(r'\toprule'+'\n')
        f.write(r'example & flux samples & eigenvalues/sample & sign changes & $\min |E|$ \\'+'\n')
        f.write(r'\midrule'+'\n')
        f.write(f"ex2 & {len(thetas)} & {vals_by_theta.shape[1]} & {crossings} & ${_tex_num(min_abs)}$ \\\\\n")
        f.write(r'\bottomrule'+'\n')
        f.write(r'\end{tabular}'+'\n')
    print('flux spectral flow:', fig_path, tex_path,
          f'crossings={crossings} min_abs={min_abs:.4g} span={max_span:.4g}')
    return fig_path, tex_path

def pl_collar_scaling():
    rows = []
    for s in PL_SCALING_LEVELS:
        try:
            v0, f0 = icosahedron()
            vS, fS = subdivide_icosphere(v0, f0, s)
            vp, fp, cycles, bV, bE, rounds = build_ex1_cap(vS, fS, z_cut=Z_CUT_CAP)
            vDp, fDp, cyclesD, boundary_vertices, copy_map, copy_only_vertices = build_double(vp, fp)
            Bp, Pp, Sp, (uDp, vDpe, edge_to_idDp), (nVdp, nEdp, nFdp) = build_B_P_S(vDp, fDp)
            Hplusp = build_Hpm_from_BS(Bp, Sp, sign=+1).tocsr()
            Brel, Prel, Srel, (up, vpe, edge_to_idp, face_edge_idsp, incp), (nVp, nEp, nFp) = build_B_P_S_relative(vp, fp)
            Hpl = build_Hpm_from_BS(Brel, Srel, sign=+1).tocsr()
            basis = original_basis_indices(nVp, nEp, nFp, nVdp, nEdp)
            Hpartial = Hplusp[basis, :][:, basis].tocsr()
            R = (Hpartial - Hpl).tocsr()
            bd = boundary_basis_open(fp, nVp, nEp)
            dist = distances_to_boundary(nVp, nEp, nFp, up, vpe, face_edge_idsp, bd)
            support_dim, support_radius, *_ = sparse_support_stats(R, dist)
            dimH = nVp+nEp+nFp
            rows.append((s, rounds, dimH, len(bd), support_dim, support_radius,
                         float(len(bd)/dimH), float(support_dim/dimH)))
        except Exception as e:
            warnings.warn(f"[PL scaling] skip level {s}: {e}")

    fig_path = None
    if rows:
        levels = [r[0] for r in rows]
        boundary_frac = [r[6] for r in rows]
        support_frac = [r[7] for r in rows]
        plt.figure(figsize=(7.6, 4.0))
        plt.plot(levels, boundary_frac, '-o', label='boundary subspace / dim')
        plt.plot(levels, support_frac, '-s', label=r'$R_{\partial,+}$ support / dim')
        plt.xlabel('icosphere subdivision level')
        plt.ylabel('fraction of original-side Hilbert space')
        plt.title('Boundary-scale collar correction under refinement')
        plt.legend()
        fig_path = savefig('fig_pl_collar_scaling.png')
    else:
        fig_path = save_placeholder_png('fig_pl_collar_scaling.png', 'PL collar scaling skipped.')

    tex_path = os.path.join(OUTDIR, 'pl_collar_scaling_table.tex')
    with open(tex_path, 'w') as f:
        f.write(r'\begin{tabular}{@{}rrrrrrr@{}}'+'\n')
        f.write(r'\toprule'+'\n')
        f.write(r'subdiv. & bary rounds & $\dim\Hil(\K)$ & boundary dim. & support dim. & radius & support/dim \\'+'\n')
        f.write(r'\midrule'+'\n')
        for s, rounds, dimH, bdim, sdim, radius, bfrac, sfrac in rows:
            f.write(f"{s} & {rounds} & {dimH} & {bdim} & {sdim} & {radius} & ${_tex_num(sfrac)}$ \\\\\n")
        f.write(r'\bottomrule'+'\n')
        f.write(r'\end{tabular}'+'\n')
    print('PL collar scaling:', fig_path, tex_path, rows)
    return fig_path, tex_path

fig_copy, tex_copy = copy_potential_robustness_ex1()
all_files.extend([fig_copy, tex_copy])
fig_pl, tex_pl = pl_boundary_comparison_ex1()
all_files.extend([fig_pl, tex_pl])
fig_flux, tex_flux = flux_spectral_flow_ex2()
all_files.extend([fig_flux, tex_flux])
fig_pl_scale, tex_pl_scale = pl_collar_scaling()
all_files.extend([fig_pl_scale, tex_pl_scale])

zip_path = write_zip_bundle(all_files)

print("\n============================================================")
print(f"Wrote outputs to: {OUTDIR}/")
print(f"Zipped into: {zip_path}")
print("============================================================\n")

print("(Local run; outputs are in paper_outputs/.)")
