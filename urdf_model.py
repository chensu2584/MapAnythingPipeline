"""URDF forward kinematics and geometry loading for G1 and G2.

G1 describes its collision geometry as sphere/cylinder/box primitives; G2 uses
STL and DAE meshes referenced through ``package://`` URIs.  Both are loaded here
into the same representation so callers do not branch on the robot.

Nothing is guessed.  A mesh that cannot be located or decoded is an error rather
than a silently skipped piece of robot, because the usual reason to ask for this
geometry is to decide which pixels are the robot itself, and a hole in that
answer is a piece of robot mistaken for the scene.

Geometry is loaded per kind on first use.  Collision and visual sets often need
different decoders -- G2's collision meshes are STL while some visual parts are
COLLADA -- and a missing optional decoder must not deny the caller the set it
actually asked for.
"""

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

JOINT_TYPES = {"fixed", "revolute", "continuous", "prismatic"}


def transform_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = xyz
    return result


def rotation_about_axis(axis, angle) -> np.ndarray:
    x, y, z = (float(v) for v in axis)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    result = np.eye(4)
    result[:3, :3] = (
        np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    )
    return result


def _vector(text, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    values = np.asarray(default if text is None else [float(v) for v in text.split()])
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"Expected three finite values, got {text!r}")
    return values.astype(np.float64)


@dataclasses.dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclasses.dataclass(frozen=True)
class Geometry:
    """One collision or visual shape, already triangulated in its link frame."""

    link: str
    kind: str
    vertices: np.ndarray
    faces: np.ndarray
    source: str


def _resolve_mesh(filename: str, mesh_roots: list[Path]) -> Path:
    """Locate a ``package://`` mesh under any of the supplied roots.

    Descriptions are not consistent about ``mesh`` versus ``meshes``, so both
    spellings are tried before giving up.
    """
    relative = filename
    if relative.startswith("package://"):
        relative = relative[len("package://") :]
        relative = relative.split("/", 1)[1] if "/" in relative else relative
    relative = relative.lstrip("/")
    candidates = [relative]
    if relative.startswith("meshes/"):
        candidates.append("mesh/" + relative[len("meshes/") :])
    elif relative.startswith("mesh/"):
        candidates.append("meshes/" + relative[len("mesh/") :])
    for root in mesh_roots:
        for candidate in candidates:
            path = root / candidate
            if path.is_file():
                return path
        # Fall back to a basename search: some descriptions nest differently.
        matches = sorted(root.rglob(Path(relative).name))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Cannot locate mesh {filename!r} under {[str(r) for r in mesh_roots]}"
    )


def _primitive_mesh(node, kind: str):
    import trimesh

    if kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=2, radius=float(node.get("radius")))
    if kind == "cylinder":
        return trimesh.creation.cylinder(
            radius=float(node.get("radius")), height=float(node.get("length")), sections=16
        )
    if kind == "box":
        return trimesh.creation.box(extents=_vector(node.get("size")))
    raise ValueError(f"Unsupported primitive {kind!r}")


class UrdfRobot:
    """URDF tree with forward kinematics and triangulated link geometry."""

    def __init__(self, path, mesh_roots=None):
        import trimesh

        self.path = Path(path).resolve()
        try:
            root = ET.parse(self.path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise ValueError(f"Cannot read URDF {self.path}: {exc}") from exc

        self.mesh_roots = [Path(r).resolve() for r in (mesh_roots or [])]
        if not self.mesh_roots:
            # Descriptions usually sit at <package>/urdf/robot.urdf with meshes
            # beside the urdf directory.
            self.mesh_roots = [self.path.parent.parent, self.path.parent]

        self.links = {str(n.get("name")) for n in root.findall("link") if n.get("name")}
        self.joints_by_child: dict[str, Joint] = {}
        self.joints_by_name: dict[str, Joint] = {}
        for node in root.findall("joint"):
            kind = str(node.get("type"))
            if kind not in JOINT_TYPES:
                continue
            parent_node, child_node = node.find("parent"), node.find("child")
            if parent_node is None or child_node is None:
                raise ValueError(f"URDF joint {node.get('name')} has no parent/child")
            origin_node = node.find("origin")
            xyz = _vector(origin_node.get("xyz") if origin_node is not None else None)
            rpy = _vector(origin_node.get("rpy") if origin_node is not None else None)
            axis_node = node.find("axis")
            axis = _vector(axis_node.get("xyz") if axis_node is not None else None, (1, 0, 0))
            norm = float(np.linalg.norm(axis))
            if kind != "fixed" and norm <= 0.0:
                raise ValueError(f"URDF joint {node.get('name')} has a zero axis")
            if norm > 0.0:
                axis = axis / norm
            joint = Joint(
                str(node.get("name")),
                kind,
                str(parent_node.get("link")),
                str(child_node.get("link")),
                transform_from_xyz_rpy(xyz, rpy),
                axis,
            )
            if joint.child in self.joints_by_child:
                raise ValueError(f"URDF link {joint.child} has multiple parent joints")
            self.joints_by_child[joint.child] = joint
            self.joints_by_name[joint.name] = joint

        # Record what each shape is; decode meshes only when a kind is used.
        self._specs: dict[str, list[dict]] = {"collision": [], "visual": []}
        self._loaded: dict[str, list[Geometry]] = {}
        for link_node in root.findall("link"):
            link = str(link_node.get("name"))
            for kind in ("collision", "visual"):
                for node in link_node.findall(kind):
                    geometry_node = node.find("geometry")
                    if geometry_node is None:
                        continue
                    origin_node = node.find("origin")
                    xyz = _vector(origin_node.get("xyz") if origin_node is not None else None)
                    rpy = _vector(origin_node.get("rpy") if origin_node is not None else None)
                    spec = {"link": link, "local": transform_from_xyz_rpy(xyz, rpy)}
                    mesh_node = geometry_node.find("mesh")
                    if mesh_node is not None:
                        spec["mesh"] = str(mesh_node.get("filename"))
                        spec["scale"] = mesh_node.get("scale")
                    else:
                        primitive = next(
                            (
                                name
                                for name in ("sphere", "cylinder", "box")
                                if geometry_node.find(name) is not None
                            ),
                            None,
                        )
                        if primitive is None:
                            continue
                        spec["primitive"] = primitive
                        spec["node"] = geometry_node.find(primitive)
                    self._specs[kind].append(spec)

    def geometry_for(self, kind: str) -> list[Geometry]:
        """Decode and cache one geometry set, in the link frame."""
        import trimesh

        if kind not in self._specs:
            raise ValueError(f"kind must be 'collision' or 'visual', got {kind!r}")
        if kind in self._loaded:
            return self._loaded[kind]
        cache: dict[Path, Any] = {}
        loaded: list[Geometry] = []
        for spec in self._specs[kind]:
            if "mesh" in spec:
                mesh_path = _resolve_mesh(spec["mesh"], self.mesh_roots)
                if mesh_path not in cache:
                    try:
                        cache[mesh_path] = trimesh.load(mesh_path, force="mesh")
                    except ImportError as exc:
                        raise ImportError(
                            f"Cannot decode {mesh_path.name} for the {kind!r} geometry of "
                            f"link {spec['link']!r}: {exc}. Install the decoder, or request "
                            "the other geometry kind if it does not need it."
                        ) from exc
                mesh = cache[mesh_path].copy()
                if spec.get("scale"):
                    mesh.apply_scale(_vector(spec["scale"]))
                source = str(mesh_path)
            else:
                mesh = _primitive_mesh(spec["node"], spec["primitive"])
                source = spec["primitive"]
            mesh.apply_transform(spec["local"])
            loaded.append(
                Geometry(
                    spec["link"],
                    kind,
                    np.asarray(mesh.vertices, dtype=np.float64),
                    np.asarray(mesh.faces, dtype=np.int64),
                    source,
                )
            )
        self._loaded[kind] = loaded
        return loaded

    def base_to_frame(self, frame: str, joint_positions: dict, base: str = "base_link") -> np.ndarray:
        chain: list[Joint] = []
        cursor = frame
        visited: set[str] = set()
        while cursor != base:
            if cursor in visited:
                raise ValueError(f"Cycle while resolving URDF frame {frame}")
            visited.add(cursor)
            joint = self.joints_by_child.get(cursor)
            if joint is None:
                raise ValueError(f"No URDF chain from {base} to {frame}; stopped at {cursor}")
            chain.append(joint)
            cursor = joint.parent
        matrix = np.eye(4)
        for joint in reversed(chain):
            matrix = matrix @ joint.origin
            value = float(joint_positions.get(joint.name, 0.0))
            if joint.kind in {"revolute", "continuous"}:
                matrix = matrix @ rotation_about_axis(joint.axis, value)
            elif joint.kind == "prismatic":
                motion = np.eye(4)
                motion[:3, 3] = joint.axis * value
                matrix = matrix @ motion
        return matrix

    def link_poses(self, joint_positions: dict, base: str = "base_link") -> dict[str, np.ndarray]:
        poses = {}
        for link in self.links:
            try:
                poses[link] = self.base_to_frame(link, joint_positions, base)
            except ValueError:
                continue  # A link outside this base's tree is simply not placed.
        return poses

    def world_geometry(self, joint_positions: dict, kind: str = "collision", links=None):
        """Return ``[(link, vertices_in_base, faces)]`` for the requested geometry."""
        poses = self.link_poses(joint_positions)
        wanted = set(links) if links else None
        result = []
        for item in self.geometry_for(kind):
            if wanted is not None and item.link not in wanted:
                continue
            pose = poses.get(item.link)
            if pose is None:
                continue
            vertices = item.vertices @ pose[:3, :3].T + pose[:3, 3]
            result.append((item.link, vertices, item.faces))
        return result
