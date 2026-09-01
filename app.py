# ==============================================================================
# 🏗️ תוכנת אנליזה מבנית אינטראקטיבית דו-ממדית (Interactive 2D Frame FEA)
# ==============================================================================

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------
# 1. מנוע שיטת הקשיחות הישירה (Direct Stiffness Method Engine)
# ------------------------------------------------------------------------------
class Material:

  def __init__(self, name: str, E: float):
    self.name = name
    self.E = float(E)  # [kN/m^2]


class Section:

  def __init__(self, name: str, A: float, I: float):
    self.name = name
    self.A = float(A)  # [m^2]
    self.I = float(I)  # [m^4]


class Node:

  def __init__(
      self,
      node_id: int,
      x: float,
      y: float,
      restraints=(False, False, False),
      label="",
  ):
    self.id = int(node_id)
    self.x = float(x)
    self.y = float(y)
    self.restraints = tuple(restraints)  # (Rx, Ry, Rz)
    self.nodal_loads = np.array([0.0, 0.0, 0.0], dtype=float)  # [Fx, Fy, Mz]
    self.label = label


class Element2D:

  def __init__(
      self,
      elem_id: int,
      start_node: Node,
      end_node: Node,
      material: Material,
      section: Section,
      releases=(False, False),
  ):
    self.id = int(elem_id)
    self.start_node = start_node
    self.end_node = end_node
    self.material = material
    self.section = section
    self.releases = tuple(releases)  # (rel_start_M, rel_end_M)

    self.wx_glob = 0.0  # [kN/m] חיובי = ימינה
    self.wy_glob = 0.0  # [kN/m] חיובי = למטה

    self.local_forces = None
    self.global_displacements = None

  @property
  def length(self) -> float:
    return np.hypot(
        self.end_node.x - self.start_node.x, self.end_node.y - self.start_node.y
    )

  @property
  def angle(self) -> float:
    return np.arctan2(
        self.end_node.y - self.start_node.y, self.end_node.x - self.start_node.x
    )

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
    k[0, 0] = k_axial
    k[0, 3] = -k_axial
    k[3, 0] = -k_axial
    k[3, 3] = k_axial

    if not rel_start and not rel_end:
      k1 = 12 * E * I / (L**3)
      k2 = 6 * E * I / (L**2)
      k3 = 4 * E * I / L
      k4 = 2 * E * I / L
      k[1, 1] = k1
      k[1, 2] = k2
      k[1, 4] = -k1
      k[1, 5] = k2
      k[2, 1] = k2
      k[2, 2] = k3
      k[2, 4] = -k2
      k[2, 5] = k4
      k[4, 1] = -k1
      k[4, 2] = -k2
      k[4, 4] = k1
      k[4, 5] = -k2
      k[5, 1] = k2
      k[5, 2] = k4
      k[5, 4] = -k2
      k[5, 5] = k3
    elif rel_start and not rel_end:
      k1 = 3 * E * I / (L**3)
      k2 = 3 * E * I / (L**2)
      k3 = 3 * E * I / L
      k[1, 1] = k1
      k[1, 4] = -k1
      k[1, 5] = k2
      k[4, 1] = -k1
      k[4, 4] = k1
      k[4, 5] = -k2
      k[5, 1] = k2
      k[5, 4] = -k2
      k[5, 5] = k3
    elif not rel_start and rel_end:
      k1 = 3 * E * I / (L**3)
      k2 = 3 * E * I / (L**2)
      k3 = 3 * E * I / L
      k[1, 1] = k1
      k[1, 2] = k2
      k[1, 4] = -k1
      k[2, 1] = k2
      k[2, 2] = k3
      k[2, 4] = -k2
      k[4, 1] = -k1
      k[4, 2] = -k2
      k[4, 4] = k1
    return k

  def transformation_matrix(self) -> np.ndarray:
    c, s = np.cos(self.angle), np.sin(self.angle)
    T = np.zeros((6, 6))
    T[0, 0] = c
    T[0, 1] = s
    T[1, 0] = -s
    T[1, 1] = c
    T[2, 2] = 1.0
    T[3, 3] = c
    T[3, 4] = s
    T[4, 3] = -s
    T[4, 4] = c
    T[5, 5] = 1.0
    return T

  def global_stiffness_matrix(self) -> np.ndarray:
    return (
        self.transformation_matrix().T
        @ self.local_stiffness_matrix()
        @ self.transformation_matrix()
    )

  def fixed_end_forces_local(self) -> np.ndarray:
    w_trans, w_axial = self.distributed_loads_local
    L = self.length
    f_fef = np.zeros(6)

    f_fef[0] = (w_axial * L) / 2.0
    f_fef[3] = (w_axial * L) / 2.0

    rel_start, rel_end = self.releases
    if not rel_start and not rel_end:
      f_fef[1] = (w_trans * L) / 2.0
      f_fef[2] = (w_trans * (L**2)) / 12.0
      f_fef[4] = (w_trans * L) / 2.0
      f_fef[5] = -(w_trans * (L**2)) / 12.0
    elif rel_start and not rel_end:
      f_fef[1] = 3.0 * w_trans * L / 8.0
      f_fef[4] = 5.0 * w_trans * L / 8.0
      f_fef[5] = -(w_trans * (L**2)) / 8.0
    elif not rel_start and rel_end:
      f_fef[1] = 5.0 * w_trans * L / 8.0
      f_fef[2] = (w_trans * (L**2)) / 8.0
      f_fef[4] = 3.0 * w_trans * L / 8.0
    else:
      f_fef[1] = (w_trans * L) / 2.0
      f_fef[4] = (w_trans * L) / 2.0

    return f_fef


class Structure2D:

  def __init__(self):
    self.nodes = {}
    self.elements = {}
    self.default_E = 200e6
    self.default_A = 0.006
    self.default_I = 8.0e-5
    self.displacements = None
    self.reactions = None

  def add_node(
      self, node_id: int, x: float, y: float, restraints=(False, False, False)
  ):
    node = Node(node_id, x, y, restraints)
    self.nodes[node_id] = node
    return node

  def add_element(
      self, elem_id: int, start_id: int, end_id: int, releases=(False, False)
  ):
    mat = Material(f"Mat_E{elem_id}", self.default_E)
    sec = Section(f"Sec_E{elem_id}", self.default_A, self.default_I)
    elem = Element2D(
        elem_id, self.nodes[start_id], self.nodes[end_id], mat, sec, releases
    )
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
      dof_indices = [
          i_start,
          i_start + 1,
          i_start + 2,
          i_end,
          i_end + 1,
          i_end + 2,
      ]

      for r_loc, r_glob in enumerate(dof_indices):
        F_equivalent[r_glob] += f_fef_glob[r_loc]
        for c_loc, c_glob in enumerate(dof_indices):
          K_global[r_glob, c_glob] += K_elem_g[r_loc, c_loc]

    F_nodal = np.zeros(num_dofs)
    for nid, node in self.nodes.items():
      idx = node_idx_map[nid] * 3
      F_nodal[idx : idx + 3] = node.nodal_loads

    F_total = F_nodal - F_equivalent
    restrained_dofs = [
        node_idx_map[nid] * 3 + d
        for nid, n in self.nodes.items()
        for d, r in enumerate(n.restraints)
        if r
    ]
    free_dofs = [dof for dof in range(num_dofs) if dof not in restrained_dofs]

    if not free_dofs or np.linalg.matrix_rank(
        K_global[np.ix_(free_dofs, free_dofs)]
    ) < len(free_dofs):
      raise ValueError(
          "המבנה אינו יציב סטטית!\n(בדוק אם חסרים סמכים או שיש עודף פרקים"
          " פנימיים)"
      )

    U_free = np.linalg.solve(
        K_global[np.ix_(free_dofs, free_dofs)], F_total[free_dofs]
    )
    U_global = np.zeros(num_dofs)
    U_global[free_dofs] = U_free
    self.displacements = U_global
    self.reactions = K_global @ U_global + F_equivalent - F_nodal

    for elem in self.elements.values():
      i_start = node_idx_map[elem.start_node.id] * 3
      i_end = node_idx_map[elem.end_node.id] * 3
      dofs = [i_start, i_start + 1, i_start + 2, i_end, i_end + 1, i_end + 2]
      elem.global_displacements = U_global[dofs]
      T = elem.transformation_matrix()
      elem.local_forces = (
          elem.local_stiffness_matrix() @ (T @ elem.global_displacements)
          + elem.fixed_end_forces_local()
      )

  def print_results(self):
    node_list = sorted(list(self.nodes.keys()))
    node_idx_map = {nid: i for i, nid in enumerate(node_list)}

    print("=" * 72)
    print("                 STRUCTURAL ANALYSIS REPORT")
    print("=" * 72)
    print("\n--- 1. NODAL DISPLACEMENTS [mm & rad] ---")
    print(
        "{:<8} {:<15} {:<15} {:<15} {:<15}".format(
            "Node", "dx [mm]", "dy [mm]", "rz [rad]", "rz [mrad]"
        )
    )
    print("-" * 70)
    for nid in node_list:
      idx = node_idx_map[nid] * 3
      dx_mm = self.displacements[idx] * 1000.0
      dy_mm = self.displacements[idx + 1] * 1000.0
      rz_rad = self.displacements[idx + 2]
      rz_mrad = rz_rad * 1000.0
      print(
          "N{:<7} {:<15.2f} {:<15.2f} {:<15.4f} {:<15.2f}".format(
              nid, dx_mm, dy_mm, rz_rad, rz_mrad
          )
      )

    print("\n--- 2. SUPPORT REACTIONS ---")
    print(
        "{:<8} {:<15} {:<15} {:<15}".format(
            "Node", "Rx [kN]", "Ry [kN]", "Mz [kNm]"
        )
    )
    print("-" * 55)
    for nid, node in self.nodes.items():
      if any(node.restraints):
        idx = node_idx_map[nid] * 3
        rx = self.reactions[idx] if node.restraints[0] else 0.0
        ry = self.reactions[idx + 1] if node.restraints[1] else 0.0
        mz = self.reactions[idx + 2] if node.restraints[2] else 0.0
        print(
            "N{:<7} {:<15.2f} {:<15.2f} {:<15.2f}".format(nid, rx, ry, mz)
        )
    print("=" * 72)


# ------------------------------------------------------------------------------
# 2. ממשק המשתמש הגרפי (Interactive Application)
# ------------------------------------------------------------------------------
class StructuralCanvasApp:
  SUPPORT_TYPES = [
      ("חופשי (Free)", (False, False, False)),
      ("סמך קבוע (Pinned)", (True, True, False)),
      ("ריתום מלא (Fixed)", (True, True, True)),
      ("סמך נייד ב-X (Roller X)", (False, True, False)),
      ("סמך נייד ב-Y (Roller Y)", (True, False, False)),
      ("ריתום נייד ב-X (Sliding X)", (False, True, True)),
      ("ריתום נייד ב-Y (Sliding Y)", (True, False, True)),
  ]

  def __init__(self, root):
    self.root = root
    self.root.title("🏗️ 2D Frame Interactive Modeler & Analyzer")
    self.root.geometry("1240x820")

    self.model = Structure2D()
    self.mode = "DRAW"
    self.grid_spacing = 50
    self.origin_x = 120
    self.origin_y = 660

    self.current_start_node = None
    self.temp_mouse_pos = None

    self._build_ui()
    self._draw_grid()

  def _build_ui(self):
    toolbar = tk.Frame(self.root, bg="#2c3e50", height=55)
    toolbar.pack(side=tk.TOP, fill=tk.X)

    self.btn_draw = tk.Button(
        toolbar,
        text="1. ✏️ שרטוט מוטות",
        bg="#3498db",
        fg="white",
        font=("Arial", 10, "bold"),
        command=lambda: self.set_mode("DRAW"),
    )
    self.btn_draw.pack(side=tk.LEFT, padx=4, pady=8)

    self.btn_prop = tk.Button(
        toolbar,
        text="2. 📐 מאפייני חתך (E,A,I)",
        bg="#95a5a6",
        fg="white",
        font=("Arial", 10, "bold"),
        command=lambda: self.set_mode("PROPERTIES"),
    )
    self.btn_prop.pack(side=tk.LEFT, padx=4, pady=8)

    self.btn_supp = tk.Button(
        toolbar,
        text="3. ⚓ סמכים ופרקים",
        bg="#95a5a6",
        fg="white",
        font=("Arial", 10, "bold"),
        command=lambda: self.set_mode("SUPPORTS"),
    )
    self.btn_supp.pack(side=tk.LEFT, padx=4, pady=8)

    self.btn_loads = tk.Button(
        toolbar,
        text="4. 🎯 עומסים",
        bg="#95a5a6",
        fg="white",
        font=("Arial", 10, "bold"),
        command=lambda: self.set_mode("LOADS"),
    )
    self.btn_loads.pack(side=tk.LEFT, padx=4, pady=8)

    self.btn_solve = tk.Button(
        toolbar,
        text="5. 🚀 פתור והצג מהלכים",
        bg="#27ae60",
        fg="white",
        font=("Arial", 10, "bold"),
        command=self.solve_and_plot,
    )
    self.btn_solve.pack(side=tk.LEFT, padx=12, pady=8)

    btn_clear = tk.Button(
        toolbar,
        text="🗑️ נקה הכל",
        bg="#e74c3c",
        fg="white",
        font=("Arial", 10),
        command=self.clear_all,
    )
    btn_clear.pack(side=tk.RIGHT, padx=10, pady=8)

    self.status_lbl = tk.Label(
        self.root,
        text="מצב שרטוט: לחץ לקביעת התחלה, משוך ולחץ לסיום מוט. (Enter לניתוק)",
        bg="#ecf0f1",
        fg="#2c3e50",
        font=("Arial", 11, "bold"),
        anchor="w",
        padx=10,
    )
    self.status_lbl.pack(side=tk.TOP, fill=tk.X)

    self.canvas = tk.Canvas(self.root, bg="#ffffff", cursor="cross")
    self.canvas.pack(fill=tk.BOTH, expand=True)

    self.canvas.bind("<Button-1>", self.on_canvas_click)
    self.canvas.bind("<Motion>", self.on_canvas_mouse_move)
    self.root.bind("<Return>", self.on_enter_pressed)
    self.root.bind("<Escape>", lambda e: self.cancel_draw())

  def world_to_screen(self, x, y):
    return (
        self.origin_x + x * self.grid_spacing,
        self.origin_y - y * self.grid_spacing,
    )

  def screen_to_world(self, sx, sy):
    wx = round((sx - self.origin_x) / self.grid_spacing)
    wy = round((self.origin_y - sy) / self.grid_spacing)
    return max(0, wx), max(0, wy)

  def set_mode(self, mode):
    self.mode = mode
    self.cancel_draw()
    buttons = {
        "DRAW": self.btn_draw,
        "PROPERTIES": self.btn_prop,
        "SUPPORTS": self.btn_supp,
        "LOADS": self.btn_loads,
    }
    for m, btn in buttons.items():
      btn.config(bg="#3498db" if m == mode else "#95a5a6")

    if mode == "DRAW":
      self.status_lbl.config(
          text=(
              "מצב שרטוט: לחץ לקביעת התחלת מוט, משוך ולחץ לסיום. הקש Enter"
              " לניתוק השרשור."
          )
      )
    elif mode == "PROPERTIES":
      self.status_lbl.config(
          text=(
              "מצב מאפייני חתך: לחץ על מוט לעדכון E, A, I שלו | או לחץ במקום"
              " ריק להגדרת ברירת מחדל לכל המבנה."
          )
      )
    elif mode == "SUPPORTS":
      self.status_lbl.config(
          text=(
              "מצב סמכים ופרקים: לחץ על צומת להחלפת סמך בסבב | לחץ על מוט"
              " להגדרת פרק (הצומת יהפוך לעיגול חלול)."
          )
      )
    elif mode == "LOADS":
      self.status_lbl.config(
          text=(
              "מצב עומסים: לחץ על צומת להזנת כוחות ומומנטים | לחץ על מוט להזנת"
              " עומס מפורס wx (אופקי) ו-wy (אנכי)."
          )
      )
    self.redraw()

  def cancel_draw(self):
    self.current_start_node = None
    self.temp_mouse_pos = None
    self.redraw()

  def on_enter_pressed(self, event=None):
    if self.mode == "DRAW":
      self.cancel_draw()

  def get_or_create_node(self, wx, wy):
    for nid, n in self.model.nodes.items():
      if abs(n.x - wx) < 1e-3 and abs(n.y - wy) < 1e-3:
        return n
    new_id = len(self.model.nodes) + 1
    return self.model.add_node(new_id, wx, wy)

  def on_canvas_click(self, event):
    wx, wy = self.screen_to_world(event.x, event.y)

    if self.mode == "DRAW":
      clicked_node = self.get_or_create_node(wx, wy)
      if self.current_start_node is None:
        self.current_start_node = clicked_node
      else:
        if self.current_start_node.id != clicked_node.id:
          new_eid = len(self.model.elements) + 1
          self.model.add_element(
              new_eid, self.current_start_node.id, clicked_node.id
          )
        self.current_start_node = clicked_node
      self.redraw()

    elif self.mode == "PROPERTIES":
      hit_elem = None
      for eid, elem in self.model.elements.items():
        sx0, sy0 = self.world_to_screen(elem.start_node.x, elem.start_node.y)
        sx1, sy1 = self.world_to_screen(elem.end_node.x, elem.end_node.y)
        mid_x, mid_y = (sx0 + sx1) / 2, (sy0 + sy1) / 2
        if np.hypot(event.x - mid_x, event.y - mid_y) < 30:
          hit_elem = elem
          break

      if hit_elem:
        e_val = simpledialog.askfloat(
            "מאפייני מוט",
            f"מודול אלסטיות E עבור מוט E{hit_elem.id} [kN/m^2]:",
            initialvalue=hit_elem.material.E,
        )
        if e_val is not None:
          a_val = simpledialog.askfloat(
              "מאפייני מוט",
              f"שטח חתך A עבור מוט E{hit_elem.id} [m^2]:",
              initialvalue=hit_elem.section.A,
          )
          if a_val is not None:
            i_val = simpledialog.askfloat(
                "מאפייני מוט",
                f"מומנט אינרציה I עבור מוט E{hit_elem.id} [m^4]:",
                initialvalue=hit_elem.section.I,
            )
            if i_val is not None:
              hit_elem.material.E = float(e_val)
              hit_elem.section.A = float(a_val)
              hit_elem.section.I = float(i_val)
      else:
        e_val = simpledialog.askfloat(
            "מאפיינים כלליים למבנה",
            "מודול אלסטיות כללי E [kN/m^2]:",
            initialvalue=self.model.default_E,
        )
        if e_val is not None:
          a_val = simpledialog.askfloat(
              "מאפיינים כלליים למבנה",
              "שטח חתך כללי A [m^2]:",
              initialvalue=self.model.default_A,
          )
          if a_val is not None:
            i_val = simpledialog.askfloat(
                "מאפיינים כלליים למבנה",
                "מומנט אינרציה כללי I [m^4]:",
                initialvalue=self.model.default_I,
            )
            if i_val is not None:
              self.model.default_E = float(e_val)
              self.model.default_A = float(a_val)
              self.model.default_I = float(i_val)
              for elem in self.model.elements.values():
                elem.material.E = float(e_val)
                elem.section.A = float(a_val)
                elem.section.I = float(i_val)
              messagebox.showinfo(
                  "עודכן", "המאפיינים הכלליים הוחלו על כל מוטות המבנה!"
              )
      self.redraw()

    elif self.mode == "SUPPORTS":
      for nid, n in self.model.nodes.items():
        sx, sy = self.world_to_screen(n.x, n.y)
        if np.hypot(event.x - sx, event.y - sy) < 18:
          cur_idx = 0
          for i, (name, rest) in enumerate(self.SUPPORT_TYPES):
            if n.restraints == rest:
              cur_idx = i
              break
          next_idx = (cur_idx + 1) % len(self.SUPPORT_TYPES)
          n.restraints = self.SUPPORT_TYPES[next_idx][1]
          self.redraw()
          return

      for eid, elem in self.model.elements.items():
        sx0, sy0 = self.world_to_screen(elem.start_node.x, elem.start_node.y)
        sx1, sy1 = self.world_to_screen(elem.end_node.x, elem.end_node.y)
        mid_x, mid_y = (sx0 + sx1) / 2, (sy0 + sy1) / 2
        if np.hypot(event.x - mid_x, event.y - mid_y) < 25:
          choice = simpledialog.askstring(
              "הגדרת פרק פנימי בצומת",
              f"בחר שחרור מומנט עבור מוט E{eid}:\n1 = פרק בצומת התחלה"
              f" (N{elem.start_node.id})\n2 = פרק בצומת סיום"
              f" (N{elem.end_node.id})\n3 = פרק בשני הקצוות (N{elem.start_node.id}"
              f" ו-N{elem.end_node.id})\n0 = ביטול פרקים (חיבור מונוליטי רציף)",
          )
          if choice == "1":
            elem.releases = (True, False)
          elif choice == "2":
            elem.releases = (False, True)
          elif choice == "3":
            elem.releases = (True, True)
          elif choice == "0":
            elem.releases = (False, False)
          self.redraw()
          return

    elif self.mode == "LOADS":
      for nid, n in self.model.nodes.items():
        sx, sy = self.world_to_screen(n.x, n.y)
        if np.hypot(event.x - sx, event.y - sy) < 18:
          fx = simpledialog.askfloat(
              "עומס צמתי - שלב 1/3",
              f"כוח אופקי Fx בצומת N{nid} [kN] (חיובי = ימינה):",
              initialvalue=n.nodal_loads[0],
          )
          if fx is not None:
            fy = simpledialog.askfloat(
                "עומס צמתי - שלב 2/3",
                f"כוח אנכי Fy בצומת N{nid} [kN] (חיובי = למעלה):",
                initialvalue=n.nodal_loads[1],
            )
            if fy is not None:
              mz = simpledialog.askfloat(
                  "עומס צמתי - שלב 3/3",
                  f"מומנט מרוכז Mz בצומת N{nid} [kNm] (חיובי = נגד השעון):",
                  initialvalue=n.nodal_loads[2],
              )
              if mz is not None:
                n.nodal_loads = np.array([fx, fy, mz], dtype=float)
          self.redraw()
          return

      for eid, elem in self.model.elements.items():
        sx0, sy0 = self.world_to_screen(elem.start_node.x, elem.start_node.y)
        sx1, sy1 = self.world_to_screen(elem.end_node.x, elem.end_node.y)
        mid_x, mid_y = (sx0 + sx1) / 2, (sy0 + sy1) / 2
        if np.hypot(event.x - mid_x, event.y - mid_y) < 25:
          wx = simpledialog.askfloat(
              "עומס מפורס - שלב 1/2",
              f"עומס מפורס אופקי wx על מוט E{eid} [kN/m] (חיובי = ימינה):",
              initialvalue=elem.wx_glob,
          )
          if wx is not None:
            wy = simpledialog.askfloat(
                "עומס מפורס - שלב 2/2",
                f"עומס מפורס אנכי wy על מוט E{eid} [kN/m] (חיובי = למטה):",
                initialvalue=elem.wy_glob,
            )
            if wy is not None:
              elem.wx_glob = float(wx)
              elem.wy_glob = float(wy)
          self.redraw()
          return

  def on_canvas_mouse_move(self, event):
    if self.mode == "DRAW" and self.current_start_node is not None:
      self.temp_mouse_pos = (event.x, event.y)
      self.redraw()

  def _draw_grid(self):
    w = self.canvas.winfo_width() or 1100
    h = self.canvas.winfo_height() or 750
    for x in range(self.origin_x % self.grid_spacing, w, self.grid_spacing):
      self.canvas.create_line(x, 0, x, h, fill="#f0f3f4", tags="grid")
    for y in range(self.origin_y % self.grid_spacing, h, self.grid_spacing):
      self.canvas.create_line(0, y, w, y, fill="#f0f3f4", tags="grid")
    self.canvas.create_line(
        self.origin_x,
        0,
        self.origin_x,
        h,
        fill="#bdc3c7",
        width=2,
        tags="grid",
    )
    self.canvas.create_line(
        0,
        self.origin_y,
        w,
        self.origin_y,
        fill="#bdc3c7",
        width=2,
        tags="grid",
    )

  def redraw(self):
    self.canvas.delete("all")
    self._draw_grid()

    if self.mode == "DRAW" and self.current_start_node and self.temp_mouse_pos:
      sx0, sy0 = self.world_to_screen(
          self.current_start_node.x, self.current_start_node.y
      )
      twx, twy = self.screen_to_world(
          self.temp_mouse_pos[0], self.temp_mouse_pos[1]
      )
      tsx, tsy = self.world_to_screen(twx, twy)
      self.canvas.create_line(
          sx0, sy0, tsx, tsy, fill="#e74c3c", width=2, dash=(4, 3)
      )
      self.canvas.create_oval(
          tsx - 4, tsy - 4, tsx + 4, tsy + 4, fill="#e74c3c"
      )

    for eid, elem in self.model.elements.items():
      sx0, sy0 = self.world_to_screen(elem.start_node.x, elem.start_node.y)
      sx1, sy1 = self.world_to_screen(elem.end_node.x, elem.end_node.y)
      self.canvas.create_line(sx0, sy0, sx1, sy1, fill="#2c3e50", width=4)

      mx, my = (sx0 + sx1) / 2, (sy0 + sy1) / 2
      self.canvas.create_text(
          mx,
          my - 14,
          text=f"E{eid} (L={elem.length:.1f}m)",
          fill="#2980b9",
          font=("Arial", 9, "bold"),
      )

      wy = elem.wy_glob
      L = elem.length
      if wy != 0 and L > 0:
        arr_h = 24 if wy > 0 else -24
        num_arr = max(int(L * 3), 4)
        roof_pts = []
        for s in np.linspace(0, 1, num_arr):
          px = sx0 + s * (sx1 - sx0)
          py = sy0 + s * (sy1 - sy0)
          ry = py - arr_h
          roof_pts.append((px, ry))
          self.canvas.create_line(
              px,
              ry,
              px,
              py,
              arrow=tk.LAST,
              fill="#16a085",
              width=1.5,
              arrowshape=(8, 10, 4),
          )
        for i in range(len(roof_pts) - 1):
          self.canvas.create_line(
              roof_pts[i][0],
              roof_pts[i][1],
              roof_pts[i + 1][0],
              roof_pts[i + 1][1],
              fill="#16a085",
              width=1.5,
              dash=(3, 2),
          )
        self.canvas.create_text(
            mx,
            my - arr_h - 10,
            text=f"wy={wy} kN/m ↓",
            fill="#16a085",
            font=("Arial", 9, "bold"),
        )

      wx = elem.wx_glob
      if wx != 0 and L > 0:
        arr_w = 24 if wx > 0 else -24
        num_arr = max(int(L * 3), 4)
        roof_pts_x = []
        for s in np.linspace(0, 1, num_arr):
          px = sx0 + s * (sx1 - sx0)
          py = sy0 + s * (sy1 - sy0)
          rx = px - arr_w
          roof_pts_x.append((rx, py))
          self.canvas.create_line(
              rx,
              py,
              px,
              py,
              arrow=tk.LAST,
              fill="#d35400",
              width=1.5,
              arrowshape=(8, 10, 4),
          )
        for i in range(len(roof_pts_x) - 1):
          self.canvas.create_line(
              roof_pts_x[i][0],
              roof_pts_x[i][1],
              roof_pts_x[i + 1][0],
              roof_pts_x[i + 1][1],
              fill="#d35400",
              width=1.5,
              dash=(3, 2),
          )
        self.canvas.create_text(
            mx - arr_w - 15,
            my + 14,
            text=f"wx={wx} kN/m →",
            fill="#d35400",
            font=("Arial", 9, "bold"),
        )

    for nid, node in self.model.nodes.items():
      sx, sy = self.world_to_screen(node.x, node.y)

      is_hinged_node = any(
          (elem.start_node.id == node.id and elem.releases[0])
          or (elem.end_node.id == node.id and elem.releases[1])
          for elem in self.model.elements.values()
      )

      if is_hinged_node:
        self.canvas.create_oval(
            sx - 7,
            sy - 7,
            sx + 7,
            sy + 7,
            fill="#ffffff",
            outline="#000000",
            width=2.8,
        )
      else:
        self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="#34495e")

      self.canvas.create_text(
          sx + 10,
          sy - 12,
          text=f"N{nid}({node.x:.0f},{node.y:.0f})",
          fill="black",
          font=("Arial", 9, "bold"),
      )

      rx, ry, rz = node.restraints
      if rx and ry and rz:
        self.canvas.create_rectangle(
            sx - 14, sy, sx + 14, sy + 6, fill="#34495e", outline="black"
        )
        for dx in np.linspace(-12, 12, 5):
          self.canvas.create_line(
              sx + dx, sy + 6, sx + dx - 5, sy + 14, fill="black", width=1.5
          )
        self.canvas.create_text(
            sx,
            sy + 22,
            text="Fixed",
            fill="#2c3e50",
            font=("Arial", 8, "bold"),
        )
      elif rx and ry and not rz:
        self.canvas.create_polygon(
            sx,
            sy,
            sx - 10,
            sy + 14,
            sx + 10,
            sy + 14,
            fill="#bdc3c7",
            outline="black",
            width=1.5,
        )
        self.canvas.create_line(
            sx - 14, sy + 14, sx + 14, sy + 14, fill="black", width=2
        )
        for dx in np.linspace(-10, 10, 4):
          self.canvas.create_line(
              sx + dx, sy + 14, sx + dx - 4, sy + 20, fill="black", width=1
          )
        self.canvas.create_text(
            sx,
            sy + 28,
            text="Pinned",
            fill="#2c3e50",
            font=("Arial", 8, "bold"),
        )
      elif not rx and ry and not rz:
        self.canvas.create_polygon(
            sx,
            sy,
            sx - 9,
            sy + 12,
            sx + 9,
            sy + 12,
            fill="#bdc3c7",
            outline="black",
            width=1.5,
        )
        self.canvas.create_oval(
            sx - 7, sy + 12, sx - 2, sy + 17, fill="white", outline="black"
        )
        self.canvas.create_oval(
            sx + 2, sy + 12, sx + 7, sy + 17, fill="white", outline="black"
        )
        self.canvas.create_line(
            sx - 13, sy + 18, sx + 13, sy + 18, fill="black", width=1.8
        )
        self.canvas.create_text(
            sx,
            sy + 28,
            text="Roller X",
            fill="#2c3e50",
            font=("Arial", 8, "bold"),
        )
      elif rx and not ry and not rz:
        self.canvas.create_polygon(
            sx,
            sy,
            sx - 12,
            sy - 9,
            sx - 12,
            sy + 9,
            fill="#bdc3c7",
            outline="black",
            width=1.5,
        )
        self.canvas.create_oval(
            sx - 17, sy - 7, sx - 12, sy - 2, fill="white", outline="black"
        )
        self.canvas.create_oval(
            sx - 17, sy + 2, sx - 12, sy + 7, fill="white", outline="black"
        )
        self.canvas.create_line(
            sx - 18, sy - 13, sx - 18, sy + 13, fill="black", width=1.8
        )
        self.canvas.create_text(
            sx - 28,
            sy,
            text="Roller Y",
            fill="#2c3e50",
            font=("Arial", 8, "bold"),
        )
      elif not rx and ry and rz:
        self.canvas.create_line(
            sx - 14, sy + 4, sx + 14, sy + 4, fill="#34495e", width=4
        )
        self.canvas.create_oval(
            sx - 8, sy + 6, sx - 3, sy + 11, fill="white", outline="black"
        )
        self.canvas.create_oval(
            sx + 3, sy + 6, sx + 8, sy + 11, fill="white", outline="black"
        )
        self.canvas.create_line(
            sx - 16, sy + 12, sx + 16, sy + 12, fill="black", width=2
        )
        self.canvas.create_text(
            sx,
            sy + 22,
            text="Sliding X",
            fill="#2c3e50",
            font=("Arial", 8, "bold"),
        )
      elif rx and not ry and rz:
        self.canvas.create_line(
            sx - 4, sy - 14, sx - 4, sy + 14, fill="#34495e", width=4
        )
        self.canvas.create_oval(
            sx - 11, sy - 8, sx - 6, sy - 3, fill="white", outline="black"
        )
        self.canvas.create_oval(
            sx - 11, sy + 3, sx - 6, sy + 8, fill="white", outline="black"
        )
        self.canvas.create_line(
            sx - 12, sy - 16, sx - 12, sy + 16, fill="black", width=2
        )
        self.canvas.create_text(
            sx - 30,
            sy,
            text="Sliding Y",
            fill="#2c3e50",
            font=("Arial", 8, "bold"),
        )

      fx, fy, mz = node.nodal_loads
      if fx != 0:
        dx = 35 if fx > 0 else -35
        self.canvas.create_line(
            sx - dx,
            sy,
            sx,
            sy,
            arrow=tk.LAST,
            fill="#e74c3c",
            width=2.5,
            arrowshape=(10, 12, 5),
        )
        self.canvas.create_text(
            sx - dx,
            sy - 10,
            text=f"Fx={fx}kN",
            fill="#e74c3c",
            font=("Arial", 8, "bold"),
        )

      if fy != 0:
        dy = 35 if fy > 0 else -35
        self.canvas.create_line(
            sx,
            sy + dy,
            sx,
            sy,
            arrow=tk.LAST,
            fill="#e74c3c",
            width=2.5,
            arrowshape=(10, 12, 5),
        )
        self.canvas.create_text(
            sx + 15,
            sy + dy / 2,
            text=f"Fy={fy}kN",
            fill="#e74c3c",
            font=("Arial", 8, "bold"),
        )

      if mz != 0:
        r_m = 18
        start_ang = 30 if mz > 0 else 150
        extent_ang = 240 if mz > 0 else -240
        self.canvas.create_arc(
            sx - r_m,
            sy - r_m,
            sx + r_m,
            sy + r_m,
            start=start_ang,
            extent=extent_ang,
            style=tk.ARC,
            outline="#8e44ad",
            width=2.5,
        )
        arrow_tip_x = sx + (r_m + 3 if mz > 0 else -(r_m + 3))
        arrow_tip_y = sy - 6
        self.canvas.create_polygon(
            arrow_tip_x,
            arrow_tip_y - 4,
            arrow_tip_x,
            arrow_tip_y + 4,
            arrow_tip_x + (4 if mz > 0 else -4),
            arrow_tip_y,
            fill="#8e44ad",
        )
        self.canvas.create_text(
            sx,
            sy - 26,
            text=f"Mz={mz}kNm",
            fill="#8e44ad",
            font=("Arial", 8, "bold"),
        )

  def solve_and_plot(self):
    if not self.model.elements:
      messagebox.showwarning("שגיאה", "אנא שרטט לפחות אלמנט אחד לפני החישוב!")
      return
    try:
      self.model.solve()
    except Exception as e:
      messagebox.showerror("שגיאה באנליזה", str(e))
      return

    self.model.print_results()
    self._show_displacements_window()
    self._plot_results_window()

  def _show_displacements_window(self):
    disp_win = tk.Toplevel(self.root)
    disp_win.title("📋 Nodal Displacements Report [mm & rad]")
    disp_win.geometry("620x360")

    lbl_title = tk.Label(
        disp_win,
        text="דוח תזוזות צמתים (תזוזות במילימטרים וסיבובים ברדיאנים)",
        font=("Arial", 11, "bold"),
        pady=10,
    )
    lbl_title.pack()

    tree = ttk.Treeview(
        disp_win,
        columns=("Node", "dx", "dy", "rz", "rz_mrad"),
        show="headings",
        height=10,
    )
    tree.heading("Node", text="צומת (Node)")
    tree.heading("dx", text="dx [mm]")
    tree.heading("dy", text="dy [mm]")
    tree.heading("rz", text="rz [rad]")
    tree.heading("rz_mrad", text="rz [mrad]")

    tree.column("Node", width=90, anchor="center")
    tree.column("dx", width=120, anchor="center")
    tree.column("dy", width=120, anchor="center")
    tree.column("rz", width=120, anchor="center")
    tree.column("rz_mrad", width=120, anchor="center")

    node_list = sorted(list(self.model.nodes.keys()))
    node_idx_map = {nid: i for i, nid in enumerate(node_list)}

    for nid in node_list:
      idx = node_idx_map[nid] * 3
      dx_mm = self.model.displacements[idx] * 1000.0
      dy_mm = self.model.displacements[idx + 1] * 1000.0
      rz_rad = self.model.displacements[idx + 2]
      rz_mrad = rz_rad * 1000.0

      tree.insert(
          "",
          tk.END,
          values=(
              f"N{nid}",
              f"{dx_mm:.2f} mm",
              f"{dy_mm:.2f} mm",
              f"{rz_rad:.4f} rad",
              f"{rz_mrad:.2f} mrad",
          ),
      )

    tree.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

    btn_close = tk.Button(
        disp_win,
        text="סגור",
        bg="#34495e",
        fg="white",
        command=disp_win.destroy,
        width=12,
    )
    btn_close.pack(pady=8)

  def _plot_results_window(self):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    self._draw_diagram_on_ax(
        axs[0, 0], "AFD", "Axial Force Diagram (AFD) [kN]", "#27ae60"
    )
    self._draw_diagram_on_ax(
        axs[0, 1], "SFD", "Shear Force Diagram (SFD) [kN]", "#0984e3"
    )
    self._draw_diagram_on_ax(
        axs[1, 0],
        "BMD",
        "Bending Moment (BMD) [kNm] - Tension Side",
        "#d63031",
    )
    self._draw_deformed_on_ax(axs[1, 1])

    plt.tight_layout()
    plt.show()

  def _draw_diagram_on_ax(self, ax, diag_type, title, color, scale=0.03):
    for elem in self.model.elements.values():
      x0, y0 = elem.start_node.x, elem.start_node.y
      x1, y1 = elem.end_node.x, elem.end_node.y
      L, th = elem.length, elem.angle
      nx, ny = -np.sin(th), np.cos(th)
      tx, ty = np.cos(th), np.sin(th)
      ax.plot([x0, x1], [y0, y1], color="#2d3436", lw=2.5)

      if elem.releases[0]:
        ax.plot(
            x0,
            y0,
            "o",
            color="white",
            markeredgecolor="black",
            markersize=7,
            mew=2,
            zorder=6,
        )
      if elem.releases[1]:
        ax.plot(
            x1,
            y1,
            "o",
            color="white",
            markeredgecolor="black",
            markersize=7,
            mew=2,
            zorder=6,
        )

      f = elem.local_forces
      w_trans, w_axial = elem.distributed_loads_local
      x_loc = np.linspace(0, L, 50)

      if diag_type == "BMD":
        vals = -f[2] + f[1] * x_loc - 0.5 * w_trans * (x_loc**2)
        offset = -vals * scale
      elif diag_type == "SFD":
        vals = f[1] - w_trans * x_loc
        offset = vals * scale
      elif diag_type == "AFD":
        vals = -f[0] + w_axial * x_loc
        offset = vals * scale

      dx_g = x0 + tx * x_loc + nx * offset
      dy_g = y0 + ty * x_loc + ny * offset
      ax.plot(dx_g, dy_g, color=color, lw=1.8)
      ax.fill(
          np.append(x0 + tx * x_loc, dx_g[::-1]),
          np.append(y0 + ty * x_loc, dy_g[::-1]),
          color=color,
          alpha=0.25,
      )
      ax.text(
          dx_g[0],
          dy_g[0],
          f"{vals[0]:.1f}",
          color=color,
          fontsize=8,
          fontweight="bold",
      )
      ax.text(
          dx_g[-1],
          dy_g[-1],
          f"{vals[-1]:.1f}",
          color=color,
          fontsize=8,
          fontweight="bold",
      )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")

  def _draw_deformed_on_ax(self, ax, mag=80.0):
    for elem in self.model.elements.values():
      x0, y0 = elem.start_node.x, elem.start_node.y
      x1, y1 = elem.end_node.x, elem.end_node.y
      ax.plot([x0, x1], [y0, y1], "k--", alpha=0.4)

      L, th, E, I = (
          elem.length,
          elem.angle,
          elem.material.E,
          elem.section.I,
      )
      w_trans, w_axial = elem.distributed_loads_local
      u1, v1, th1, u2, v2, th2 = (
          elem.transformation_matrix() @ elem.global_displacements
      )
      x_loc = np.linspace(0, L, 50)
      xi = x_loc / L
      N1, N2 = 1 - 3 * xi**2 + 2 * xi**3, L * (xi - 2 * xi**2 + xi**3)
      N3, N4 = 3 * xi**2 - 2 * xi**3, L * (-xi**2 + xi**3)
      ux = (1 - xi) * u1 + xi * u2
      vy = N1 * v1 + N2 * th1 + N3 * v2 + N4 * th2
      if w_trans != 0 and E > 0 and I > 0:
        vy += -(w_trans / (24.0 * E * I)) * (x_loc**2) * ((L - x_loc) ** 2)

      c, s = np.cos(th), np.sin(th)
      xd = (x0 + c * x_loc) + (c * ux - s * vy) * mag
      yd = (y0 + s * x_loc) + (s * ux + c * vy) * mag
      ax.plot(xd, yd, color="#e84393", lw=2.5)

      if elem.releases[0]:
        ax.plot(
            xd[0],
            yd[0],
            "o",
            color="white",
            markeredgecolor="#8e44ad",
            markersize=6,
            mew=2,
            zorder=6,
        )
      if elem.releases[1]:
        ax.plot(
            xd[-1],
            yd[-1],
            "o",
            color="white",
            markeredgecolor="#8e44ad",
            markersize=6,
            mew=2,
            zorder=6,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(
        f"Deformed Shape & Curvature (x{mag})", fontsize=11, fontweight="bold"
    )

  def clear_all(self):
    if messagebox.askyesno(
        "ניקוי מודל", "האם אתה בטוח שברצונך למחוק את כל המבנה?"
    ):
      self.model = Structure2D()
      self.cancel_draw()


if __name__ == "__main__":
  root = tk.Tk()
  app = StructuralCanvasApp(root)
  root.mainloop()
