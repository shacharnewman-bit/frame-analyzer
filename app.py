import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from streamlit_drawable_canvas import st_canvas
import streamlit as st

st.set_page_config(page_title="2D Frame FEA Analyzer", layout="wide")

# --- 1. מנוע החישוב האלגברי (FEA Engine) ---
class Material:
    def __init__(self, name: str, E: float):
        self.name = name
        self.E = float(E)

class Section:
    def __init__(self, name: str, A: float, I: float):
        self.name = name
        self.A = float(A)
        self.I = float(I)

class Node:
    def __init__(self, node_id: int, x: float, y: float, restraints=(False, False, False)):
        self.id = int(node_id)
        self.x = float(x)
        self.y = float(y)
        self.restraints = tuple(restraints)
        self.nodal_loads = np.array([0.0, 0.0, 0.0], dtype=float)

class Element2D:
    def __init__(self, elem_id: int, start_node: Node, end_node: Node, material: Material, section: Section, releases=(False, False)):
        self.id = int(elem_id)
        self.start_node = start_node
        self.end_node = end_node
        self.material = material
        self.section = section
        self.releases = tuple(releases)
        self.wx_glob = 0.0
        self.wy_glob = 0.0
        self.local_forces = None
        self.global_displacements = None

    @property
    def length(self) -> float:
        return np.hypot(self.end_node.x - self.start_node.x, self.end_node.y - self.start_node.y)

    @property
    def angle(self) -> float:
        return np.arctan2(self.end_node.y - self.start_node.y, self.end_node.x - self.start_node.x)

    @property
    def distributed_loads_local(self):
        th = self.angle
        w_trans = self.wx_glob * np.sin(th) + self.wy_glob * np.cos(th)
        w_axial = self.wx_glob * np.cos(th) - self.wy_glob * np.sin(th)
        return w_trans, w_axial

    def local_stiffness_matrix(self) -> np.ndarray:
        E, A, I, L = self.material.E, self.section.A, self.section.I, self.length
        rel_start, rel_end = self.releases
        k = np.zeros((6, 6))
        k_axial = (E * A) / L
        k[0, 0] = k_axial;  k[0, 3] = -k_axial
        k[3, 0] = -k_axial; k[3, 3] = k_axial

        if not rel_start and not rel_end:
            k1 = 12 * E * I / (L**3); k2 = 6 * E * I / (L**2); k3 = 4 * E * I / L; k4 = 2 * E * I / L
            k[1, 1] = k1;  k[1, 2] = k2;  k[1, 4] = -k1; k[1, 5] = k2
            k[2, 1] = k2;  k[2, 2] = k3;  k[2, 4] = -k2; k[2, 5] = k4
            k[4, 1] = -k1; k[4, 2] = -k2; k[4, 4] = k1;  k[4, 5] = -k2
            k[5, 1] = k2;  k[5, 2] = k4;  k[5, 4] = -k2; k[5, 5] = k3
        elif rel_start and not rel_end:
            k1 = 3 * E * I / (L**3); k2 = 3 * E * I / (L**2); k3 = 3 * E * I / L
            k[1, 1] = k1;  k[1, 4] = -k1; k[1, 5] = k2
            k[4, 1] = -k1; k[4, 4] = k1;  k[4, 5] = -k2
            k[5, 1] = k2;  k[5, 4] = -k2; k[5, 5] = k3
        elif not rel_start and rel_end:
            k1 = 3 * E * I / (L**3); k2 = 3 * E * I / (L**2); k3 = 3 * E * I / L
            k[1, 1] = k1;  k[1, 2] = k2;  k[1, 4] = -k1
            k[2, 1] = k2;  k[2, 2] = k3;  k[2, 4] = -k2
            k[4, 1] = -k1; k[4, 2] = -k2; k[4, 4] = k1
        return k

    def transformation_matrix(self) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        T = np.zeros((6, 6))
        T[0, 0] = c;  T[0, 1] = s
        T[1, 0] = -s; T[1, 1] = c
        T[2, 2] = 1.0
        T[3, 3] = c;  T[3, 4] = s
        T[4, 3] = -s; T[4, 4] = c
        T[5, 5] = 1.0
        return T

    def global_stiffness_matrix(self) -> np.ndarray:
        return self.transformation_matrix().T @ self.local_stiffness_matrix() @ self.transformation_matrix()

    def fixed_end_forces_local(self) -> np.ndarray:
        w_trans, w_axial = self.distributed_loads_local
        L = self.length
        f_fef = np.zeros(6)
        f_fef[0] = (w_axial * L) / 2.0
        f_fef[3] = (w_axial * L) / 2.0
        rel_start, rel_end = self.releases
        if not rel_start and not rel_end:
            f_fef[1] = (w_trans * L) / 2.0;       f_fef[2] = (w_trans * (L**2)) / 12.0
            f_fef[4] = (w_trans * L) / 2.0;       f_fef[5] = -(w_trans * (L**2)) / 12.0
        elif rel_start and not rel_end:
            f_fef[1] = 3.0 * w_trans * L / 8.0;   f_fef[4] = 5.0 * w_trans * L / 8.0
            f_fef[5] = -(w_trans * (L**2)) / 8.0
        elif not rel_start and rel_end:
            f_fef[1] = 5.0 * w_trans * L / 8.0;   f_fef[2] = (w_trans * (L**2)) / 8.0
            f_fef[4] = 3.0 * w_trans * L / 8.0
        else:
            f_fef[1] = (w_trans * L) / 2.0;       f_fef[4] = (w_trans * L) / 2.0
        return f_fef

class Structure2D:
    def __init__(self):
        self.nodes = {}
        self.elements = {}
        self.displacements = None
        self.reactions = None

    def add_node(self, node_id: int, x: float, y: float, restraints=(False, False, False)):
        node = Node(node_id, x, y, restraints)
        self.nodes[node_id] = node
        return node

    def add_element(self, elem_id: int, start_id: int, end_id: int, E: float, A: float, I: float, releases=(False, False)):
        mat = Material(f"Mat_{elem_id}", E)
        sec = Section(f"Sec_{elem_id}", A, I)
        elem = Element2D(elem_id, self.nodes[start_id], self.nodes[end_id], mat, sec, releases)
        self.elements[elem_id] = elem
        return elem

    def solve(self):
        node_list = sorted(list(self.nodes.keys()))
        node_idx_map = {nid: i for i, nid in enumerate(node_list)}
        num_dofs = len(node_list) * 3
        K_global = np.zeros((num_dofs, num_dofs))
        F_equivalent = np.zeros(num_dofs)

        for elem in self.elements.values():
            K_elem_g = elem.global_stiffness_matrix()
            T = elem.transformation_matrix()
            f_fef_glob = T.T @ elem.fixed_end_forces_local()
            i_start = node_idx_map[elem.start_node.id] * 3
            i_end = node_idx_map[elem.end_node.id] * 3
            dofs = [i_start, i_start+1, i_start+2, i_end, i_end+1, i_end+2]
            for r_loc, r_glob in enumerate(dofs):
                F_equivalent[r_glob] += f_fef_glob[r_loc]
                for c_loc, c_glob in enumerate(dofs):
                    K_global[r_glob, c_glob] += K_elem_g[r_loc, c_loc]

        F_nodal = np.zeros(num_dofs)
        for nid, node in self.nodes.items():
            idx = node_idx_map[nid] * 3
            F_nodal[idx:idx+3] = node.nodal_loads

        F_total = F_nodal - F_equivalent
        restrained_dofs = [node_idx_map[nid]*3 + d for nid, n in self.nodes.items() for d, r in enumerate(n.restraints) if r]
        free_dofs = [dof for dof in range(num_dofs) if dof not in restrained_dofs]

        if not free_dofs or np.linalg.matrix_rank(K_global[np.ix_(free_dofs, free_dofs)]) < len(free_dofs):
            raise ValueError("המבנה אינו יציב סטטית (חסרים סמכים או עודף פרקים פנימיים)!")

        U_free = np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], F_total[free_dofs])
        U_global = np.zeros(num_dofs)
        U_global[free_dofs] = U_free
        self.displacements = U_global
        self.reactions = K_global @ U_global + F_equivalent - F_nodal

        for elem in self.elements.values():
            i_start = node_idx_map[elem.start_node.id] * 3
            i_end = node_idx_map[elem.end_node.id] * 3
            dofs = [i_start, i_start+1, i_start+2, i_end, i_end+1, i_end+2]
            elem.global_displacements = U_global[dofs]
            T = elem.transformation_matrix()
            elem.local_forces = elem.local_stiffness_matrix() @ (T @ elem.global_displacements) + elem.fixed_end_forces_local()


# --- 2. ממשק המשתמש (Streamlit Web UI) ---
st.title("🏗️ 2D Frame FEA Analyzer - ממשק רשת אינטראקטיבי")

# אתחול Session State למודל
if "elements_data" not in st.session_state:
    st.session_state.elements_data = [] # [(x1,y1, x2,y2)]
if "nodes_dict" not in st.session_state:
    st.session_state.nodes_dict = {}

col_canvas, col_ctrl = st.columns([3, 2])

with col_canvas:
    st.subheader("1. שרטוט מוטות על גבי הגריד")
    st.caption("בחר כלי 'Line' מהתפריט שמתחת ללוח ומתח מוטות מנקודה לנקודה (כל משבצת = 1 מטר)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=4,
        stroke_color="#2c3e50",
        background_color="#ffffff",
        height=500,
        width=700,
        drawing_mode="line",
        key="canvas",
    )

GRID_SIZE = 50 # 50 פיקסלים = 1 מטר
ORIGIN_X, ORIGIN_Y = 50, 450

def snap_coord(px, py):
    x_m = round((px - ORIGIN_X) / GRID_SIZE)
    y_m = round((ORIGIN_Y - py) / GRID_SIZE)
    return max(0, x_m), max(0, y_m)

# חילוץ אלמנטים מהקנבס
current_elements = []
nodes_set = {}
if canvas_result.json_data is not None:
    objects = canvas_result.json_data["objects"]
    for obj in objects:
        if obj["type"] == "line":
            x1, y1 = snap_coord(obj["left"], obj["top"])
            x2, y2 = snap_coord(obj["left"] + obj["width"], obj["top"] + obj["height"])
            if (x1, y1) != (x2, y2):
                current_elements.append(((x1, y1), (x2, y2)))
                nodes_set[(x1, y1)] = True
                nodes_set[(x2, y2)] = True

with col_ctrl:
    st.subheader("2. הגדרת סמכים, חתכים ועומסים")
    
    # מאפייני חתך כלליים
    with st.expander("📐 מאפייני חתך וחומר כלליים", expanded=False):
        def_E = st.number_input("מודול אלסטיות E [kN/m²]", value=200e6, format="%.0e")
        def_A = st.number_input("שטח חתך A [m²]", value=0.006, format="%.4f")
        def_I = st.number_input("מומנט אינרציה I [m⁴]", value=8.0e-5, format="%.6f")

    # הגדרת סמכים לצמתים
    supp_options = {
        "חופשי": (False, False, False),
        "סמך קבוע (Pinned)": (True, True, False),
        "ריתום מלא (Fixed)": (True, True, True),
        "סמך נייד אופקי (Roller X)": (False, True, False),
        "סמך נייד אנכי (Roller Y)": (True, False, False),
        "ריתום נייד ב-X": (False, True, True),
        "ריתום נייד ב-Y": (True, False, True),
    }
    
    node_configs = {}
    if nodes_set:
        with st.expander("⚓ סמכים ועומסים צמתיים", expanded=True):
            for i, (nx, ny) in enumerate(sorted(nodes_set.keys()), start=1):
                st.markdown(f"**צומת N{i} במיקום ({nx}, {ny})**")
                c1, c2 = st.columns(2)
                with c1:
                    s_type = st.selectbox(f"סמך בצומת N{i}", list(supp_options.keys()), key=f"supp_{i}")
                with c2:
                    fx = st.number_input(f"Fx [kN] N{i}", value=0.0, key=f"fx_{i}")
                    fy = st.number_input(f"Fy [kN] N{i}", value=0.0, key=f"fy_{i}")
                    mz = st.number_input(f"Mz [kNm] N{i}", value=0.0, key=f"mz_{i}")
                node_configs[(nx, ny)] = {"id": i, "restraints": supp_options[s_type], "loads": [fx, fy, mz]}

    # הגדרת עומסים ופרקים למוטות
    elem_configs = []
    if current_elements:
        with st.expander("🎯 עומסים ופרקים למוטות", expanded=False):
            for i, ((x1, y1), (x2, y2)) in enumerate(current_elements, start=1):
                st.markdown(f"**מוט E{i}: ({x1},{y1}) -> ({x2},{y2})**")
                c1, c2 = st.columns(2)
                with c1:
                    rel_choice = st.selectbox(f"פרקים פנימיים E{i}", ["ללא (רציף)", "פרק בהתחלה", "פרק בסוף", "פרק בשני הקצוות"], key=f"rel_{i}")
                    rel_map = {"ללא (רציף)": (False, False), "פרק בהתחלה": (True, False), "פרק בסוף": (False, True), "פרק בשני הקצוות": (True, True)}
                with c2:
                    wx = st.number_input(f"wx [kN/m] (ימינה) E{i}", value=0.0, key=f"wx_{i}")
                    wy = st.number_input(f"wy [kN/m] (למטה) E{i}", value=0.0, key=f"wy_{i}")
                elem_configs.append({"id": i, "releases": rel_map[rel_choice], "wx": wx, "wy": wy})

# --- 3. הרצה והפקת תוצאות ---
if st.button("🚀 פתור מבנה והצג מהלכים", type="primary", use_container_width=True):
    if not current_elements:
        st.error("אנא שרטט לפחות אלמנט אחד על גבי הגריד!")
    else:
        model = Structure2D()
        for (nx, ny), data in node_configs.items():
            n = model.add_node(data["id"], nx, ny, data["restraints"])
            n.nodal_loads = np.array(data["loads"], dtype=float)

        for i, (((x1, y1), (x2, y2)), e_cfg) in enumerate(zip(current_elements, elem_configs), start=1):
            n1_id = node_configs[(x1, y1)]["id"]
            n2_id = node_configs[(x2, y2)]["id"]
            elem = model.add_element(i, n1_id, n2_id, def_E, def_A, def_I, e_cfg["releases"])
            elem.wx_glob = e_cfg["wx"]
            elem.wy_glob = e_cfg["wy"]

        try:
            model.solve()
            st.success("האנליזה הושלמה בהצלחה!")

            # טבלת תזוזות
            st.subheader("📋 דוח תזוזות צמתים [mm]")
            disp_data = []
            node_list = sorted(list(model.nodes.keys()))
            node_idx_map = {nid: idx for idx, nid in enumerate(node_list)}
            for nid in node_list:
                idx = node_idx_map[nid] * 3
                disp_data.append({
                    "צומת": f"N{nid}",
                    "dx [mm]": f"{model.displacements[idx]*1000:.2f}",
                    "dy [mm]": f"{model.displacements[idx+1]*1000:.2f}",
                    "rz [rad]": f"{model.displacements[idx+2]:.4f}",
                    "rz [mrad]": f"{model.displacements[idx+2]*1000:.2f}",
                })
            st.table(disp_data)

            # שרטוט 4 הדיאגרמות
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            
            def draw_diag(ax, d_type, title, color, scale=0.03):
                for elem in model.elements.values():
                    x0, y0 = elem.start_node.x, elem.start_node.y
                    x1, y1 = elem.end_node.x, elem.end_node.y
                    L, th = elem.length, elem.angle
                    nx, ny = -np.sin(th), np.cos(th)
                    tx, ty = np.cos(th), np.sin(th)
                    ax.plot([x0, x1], [y0, y1], color="#2d3436", lw=2)
                    if elem.releases[0]: ax.plot(x0, y0, "o", color="white", markeredgecolor="black", markersize=6, zorder=5)
                    if elem.releases[1]: ax.plot(x1, y1, "o", color="white", markeredgecolor="black", markersize=6, zorder=5)

                    f = elem.local_forces
                    w_trans, w_axial = elem.distributed_loads_local
                    x_loc = np.linspace(0, L, 40)
                    if d_type == "BMD": vals = -f[2] + f[1]*x_loc - 0.5*w_trans*(x_loc**2); offset = -vals * scale
                    elif d_type == "SFD": vals = f[1] - w_trans*x_loc; offset = vals * scale
                    elif d_type == "AFD": vals = -f[0] + w_axial*x_loc; offset = vals * scale

                    dx_g = x0 + tx*x_loc + nx*offset
                    dy_g = y0 + ty*x_loc + ny*offset
                    ax.plot(dx_g, dy_g, color=color, lw=1.6)
                    ax.fill(np.append(x0 + tx*x_loc, dx_g[::-1]), np.append(y0 + ty*x_loc, dy_g[::-1]), color=color, alpha=0.2)
                    ax.text(dx_g[0], dy_g[0], f"{vals[0]:.1f}", color=color, fontsize=7, fontweight="bold")
                    ax.text(dx_g[-1], dy_g[-1], f"{vals[-1]:.1f}", color=color, fontsize=7, fontweight="bold")
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.set_title(title, fontsize=10, fontweight="bold")

            draw_diag(axs[0, 0], "AFD", "Axial Force Diagram (AFD) [kN]", "#27ae60")
            draw_diag(axs[0, 1], "SFD", "Shear Force Diagram (SFD) [kN]", "#0984e3")
            draw_diag(axs[1, 0], "BMD", "Bending Moment (BMD) [kNm] - Tension Side", "#d63031")

            # דפורמציה
            ax_def = axs[1, 1]
            for elem in model.elements.values():
                x0, y0 = elem.start_node.x, elem.start_node.y
                x1, y1 = elem.end_node.x, elem.end_node.y
                ax_def.plot([x0, x1], [y0, y1], "k--", alpha=0.3)
                L, th, E, I = elem.length, elem.angle, elem.material.E, elem.section.I
                w_trans, _ = elem.distributed_loads_local
                u1, v1, th1, u2, v2, th2 = elem.transformation_matrix() @ elem.global_displacements
                x_loc = np.linspace(0, L, 40)
                xi = x_loc / L
                N1, N2 = 1 - 3*xi**2 + 2*xi**3, L*(xi - 2*xi**2 + xi**3)
                N3, N4 = 3*xi**2 - 2*xi**3, L*(-xi**2 + xi**3)
                ux = (1 - xi)*u1 + xi*u2
                vy = N1*v1 + N2*th1 + N3*v2 + N4*th2
                if w_trans != 0 and E > 0 and I > 0: vy += -(w_trans / (24.0 * E * I)) * (x_loc**2) * ((L - x_loc)**2)
                c, s = np.cos(th), np.sin(th)
                xd = (x0 + c*x_loc) + (c*ux - s*vy)*80.0
                yd = (y0 + s*x_loc) + (s*ux + c*vy)*80.0
                ax_def.plot(xd, yd, color="#e84393", lw=2)

            ax_def.set_aspect("equal", adjustable="datalim")
            ax_def.grid(True, linestyle=":", alpha=0.5)
            ax_def.set_title("Deformed Shape (x80)", fontsize=10, fontweight="bold")

            st.pyplot(fig)

        except Exception as err:
            st.error(f"שגיאה באנליזה: {err}")
