import os
import sys
import json
import random
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ===============================================Ы==============================
# Класс Module
# =============================================================================
@dataclass
class Module:
    """
    Класс Module описывает один электронный модуль, предназначенный
    для размещения в отсеке фюзеляжа.
    """
    id: str
    name: str
    width_mm: float
    height_mm: float
    depth_mm: float
    mass_kg: float
    thermal_w: float
    emc_threshold_v_m: float
    role: str
    position_mm: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    color: Tuple[float, float, float] = field(
        default_factory=lambda: (random.random(), random.random(), random.random()))
    box_corners: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        valid_roles = {'emitter', 'receiver', 'neutral'}
        if self.role not in valid_roles:
            raise ValueError(f"Неверная роль модуля {self.id}: {self.role}. Допустимые значения: {valid_roles}.")
        if self.width_mm <= 0 or self.height_mm <= 0 or self.depth_mm <= 0:
            raise ValueError(f"Габариты модуля {self.id} должны быть положительными числами.")
        if self.mass_kg <= 0:
            raise ValueError(f"Масса модуля {self.id} должна быть положительной.")
        if self.thermal_w < 0:
            raise ValueError(f"Тепловыделение модуля {self.id} не может быть отрицательным.")
        if self.emc_threshold_v_m < 0:
            raise ValueError(f"Порог ЭМС модуля {self.id} не может быть отрицательным.")

    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        x_center, y_center, z_center = self.position_mm
        half_x = self.width_mm / 2.0
        half_y = self.height_mm / 2.0
        half_z = self.depth_mm / 2.0
        x_min = x_center - half_x
        x_max = x_center + half_x
        y_min = y_center - half_y
        y_max = y_center + half_y
        z_min = z_center - half_z
        z_max = z_center + half_z
        return x_min, x_max, y_min, y_max, z_min, z_max

    def set_random_position(self, compartment: 'Compartment'):
        x_min_allowed = self.width_mm / 2.0
        x_max_allowed = compartment.width_mm - self.width_mm / 2.0
        y_min_allowed = self.height_mm / 2.0
        y_max_allowed = compartment.height_mm - self.height_mm / 2.0
        z_min_allowed = self.depth_mm / 2.0
        z_max_allowed = compartment.depth_mm - self.depth_mm / 2.0
        x = random.uniform(x_min_allowed, x_max_allowed)
        y = random.uniform(y_min_allowed, y_max_allowed)
        z = random.uniform(z_min_allowed, z_max_allowed)
        self.position_mm = (x, y, z)

    def update_box_corners(self):
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_bounds()
        self.box_corners = [
            (x_min, y_min, z_min),
            (x_min, y_min, z_max),
            (x_min, y_max, z_min),
            (x_min, y_max, z_max),
            (x_max, y_min, z_min),
            (x_max, y_min, z_max),
            (x_max, y_max, z_min),
            (x_max, y_max, z_max),
        ]

# =============================================================================
# Класс Compartment
# =============================================================================
@dataclass
class Compartment:
    width_mm: float
    height_mm: float
    depth_mm: float
    max_mass_kg: float
    max_thermal_w: float
    desired_com_mm: Tuple[float, float, float]
    emc_field_v_m: float

    def __post_init__(self):
        if self.width_mm <= 0 or self.height_mm <= 0 or self.depth_mm <= 0:
            raise ValueError("Размеры отсека должны быть положительными числами.")
        if self.max_mass_kg <= 0:
            raise ValueError("Максимальная масса отсека должна быть положительной.")
        if self.max_thermal_w < 0:
            raise ValueError("Максимальное тепловыделение не может быть отрицательным.")
        if self.emc_field_v_m < 0:
            raise ValueError("Напряжённость ЭП не может быть отрицательной.")
        x0, y0, z0 = self.desired_com_mm
        if not (0 <= x0 <= self.width_mm and 0 <= y0 <= self.height_mm and 0 <= z0 <= self.depth_mm):
            raise ValueError("Проектный центр масс должен находиться внутри границ отсека.")

    def is_within_bounds(self, module: Module) -> bool:
        x_min, x_max, y_min, y_max, z_min, z_max = module.get_bounds()
        if x_min < 0 or x_max > self.width_mm:
            return False
        if y_min < 0 or y_max > self.height_mm:
            return False
        if z_min < 0 or z_max > self.depth_mm:
            return False
        return True

# =============================================================================
# Класс EMCChecker
# =============================================================================
class EMCChecker:
    def __init__(self, min_distance_mm: float):
        if min_distance_mm < 0:
            raise ValueError("Минимальное расстояние для ЭМС не может быть отрицательным.")
        self.min_distance_mm = min_distance_mm

    def check_emc_between_modules(self, modules: List[Module]) -> List[Tuple[str, str]]:
        violations = []
        n = len(modules)
        for i in range(n):
            for j in range(i + 1, n):
                mi = modules[i]
                mj = modules[j]
                if (mi.role == 'emitter' and mj.role == 'receiver') or \
                   (mi.role == 'receiver' and mj.role == 'emitter'):
                    dist = math.sqrt(
                        (mi.position_mm[0] - mj.position_mm[0]) ** 2 +
                        (mi.position_mm[1] - mj.position_mm[1]) ** 2 +
                        (mi.position_mm[2] - mj.position_mm[2]) ** 2
                    )
                    if dist < self.min_distance_mm:
                        if mi.role == 'emitter':
                            violations.append((mi.id, mj.id))
                        else:
                            violations.append((mj.id, mi.id))
        return violations

    def check_emc_field(self, module: Module, compartment: Compartment) -> bool:
        if compartment.emc_field_v_m > module.emc_threshold_v_m:
            return False
        return True

# =============================================================================
# Класс ThermalModel
# =============================================================================
class ThermalModel:
    @staticmethod
    def total_thermal(modules: List[Module]) -> float:
        total = 0.0
        for m in modules:
            total += m.thermal_w
        return total

    @staticmethod
    def check_thermal_constraint(modules: List[Module], compartment: Compartment) -> bool:
        return ThermalModel.total_thermal(modules) <= compartment.max_thermal_w

# =============================================================================
# Класс FitnessFunction
# =============================================================================
class FitnessFunction:
    def __init__(self, min_distance_mm: float):
        self.emc_checker = EMCChecker(min_distance_mm)

    @staticmethod
    def compute_center_of_mass(modules: List[Module]) -> Tuple[float, float, float]:
        total_mass = sum(m.mass_kg for m in modules)
        if total_mass == 0:
            return (0.0, 0.0, 0.0)
        x_cm = sum(m.mass_kg * m.position_mm[0] for m in modules) / total_mass
        y_cm = sum(m.mass_kg * m.position_mm[1] for m in modules) / total_mass
        z_cm = sum(m.mass_kg * m.position_mm[2] for m in modules) / total_mass
        return x_cm, y_cm, z_cm

    def compute_center_of_mass_deviation(self, modules: List[Module], desired_com: Tuple[float, float, float]) -> float:
        x_cm, y_cm, z_cm = self.compute_center_of_mass(modules)
        dx = x_cm - desired_com[0]
        dy = y_cm - desired_com[1]
        dz = z_cm - desired_com[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def check_mass_constraint(modules: List[Module], compartment: Compartment) -> bool:
        total_mass = sum(m.mass_kg for m in modules)
        return total_mass <= compartment.max_mass_kg

    @staticmethod
    def check_geometric_constraints(modules: List[Module], compartment: Compartment) -> bool:
        for mod in modules:
            if not compartment.is_within_bounds(mod):
                return False
        n = len(modules)
        for i in range(n):
            for j in range(i + 1, n):
                mi = modules[i]
                mj = modules[j]
                x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = mi.get_bounds()
                x2_min, x2_max, y2_min, y2_max, z2_min, z2_max = mj.get_bounds()
                overlap_x = not (x1_max <= x2_min or x2_max <= x1_min)
                overlap_y = not (y1_max <= y2_min or y2_max <= y1_min)
                overlap_z = not (z1_max <= z2_min or z2_max <= z1_min)
                if overlap_x and overlap_y and overlap_z:
                    return False
        return True

    def check_emc_constraints(self, modules: List[Module], compartment: Compartment) -> bool:
        for mod in modules:
            if not self.emc_checker.check_emc_field(mod, compartment):
                return False
        violations = self.emc_checker.check_emc_between_modules(modules)
        return len(violations) == 0

    def evaluate(self, modules: List[Module], compartment: Compartment) -> float:
        if not self.check_mass_constraint(modules, compartment):
            return 0.0
        if not ThermalModel.check_thermal_constraint(modules, compartment):
            return 0.0
        if not self.check_geometric_constraints(modules, compartment):
            return 0.0
        if not self.check_emc_constraints(modules, compartment):
            return 0.0
        F = self.compute_center_of_mass_deviation(modules, compartment.desired_com_mm)
        return 1.0 / (1.0 + F)

# =============================================================================
# Класс GeneticAlgorithm
# =============================================================================
class GeneticAlgorithm:
    def __init__(
            self,
            modules_template: List[Module],
            compartment: Compartment,
            population_size: int = 50,
            generations: int = 200,
            crossover_prob: float = 0.8,
            mutation_prob: float = 0.2,
            elitism_size: int = 2,
            min_emc_distance_mm: float = 50.0,
            mutation_step_mm: float = 5.0,
    ):
        if population_size <= 0:
            raise ValueError("Размер популяции должен быть положительным.")
        if generations <= 0:
            raise ValueError("Количество поколений должно быть положительным.")
        if not (0.0 < crossover_prob <= 1.0):
            raise ValueError("Вероятность скрещивания должна быть в диапазоне (0, 1].")
        if not (0.0 <= mutation_prob <= 1.0):
            raise ValueError("Вероятность мутации должна быть в диапазоне [0, 1].")
        if elitism_size < 0 or elitism_size > population_size:
            raise ValueError("Размер элитизма должен быть неотрицательным и не превосходить размер популяции.")
        if min_emc_distance_mm < 0:
            raise ValueError("Минимальное расстояние для ЭМС не может быть отрицательным.")
        if mutation_step_mm < 0:
            raise ValueError("Шаг мутации не может быть отрицательным.")
        self.modules_template = modules_template
        self.compartment = compartment
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_size = elitism_size
        self.min_emc_distance_mm = min_emc_distance_mm
        self.mutation_step_mm = mutation_step_mm
        self.fitness_function = FitnessFunction(self.min_emc_distance_mm)
        self.population: List[List[Module]] = []
        self.fitness_values: List[float] = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            individual = []
            for template in self.modules_template:
                cloned = Module(
                    id=template.id,
                    name=template.name,
                    width_mm=template.width_mm,
                    height_mm=template.height_mm,
                    depth_mm=template.depth_mm,
                    mass_kg=template.mass_kg,
                    thermal_w=template.thermal_w,
                    emc_threshold_v_m=template.emc_threshold_v_m,
                    role=template.role,
                    position_mm=(0.0, 0.0, 0.0),
                    color=template.color,
                )
                cloned.set_random_position(self.compartment)
                individual.append(cloned)
            self.population.append(individual)

    def evaluate_population(self):
        self.fitness_values = []
        for individual in self.population:
            fitness = self.fitness_function.evaluate(individual, self.compartment)
            self.fitness_values.append(fitness)

    def select_parents(self) -> List[List[Module]]:
        parents = []
        num_tournaments = self.population_size
        tournament_size = 2
        for _ in range(num_tournaments):
            indices = random.sample(range(self.population_size), tournament_size)
            best_idx = indices[0]
            for idx in indices[1:]:
                if self.fitness_values[idx] > self.fitness_values[best_idx]:
                    best_idx = idx
            parent = []
            for mod in self.population[best_idx]:
                cloned = Module(
                    id=mod.id,
                    name=mod.name,
                    width_mm=mod.width_mm,
                    height_mm=mod.height_mm,
                    depth_mm=mod.depth_mm,
                    mass_kg=mod.mass_kg,
                    thermal_w=mod.thermal_w,
                    emc_threshold_v_m=mod.emc_threshold_v_m,
                    role=mod.role,
                    position_mm=mod.position_mm,
                    color=mod.color,
                )
                parent.append(cloned)
            parents.append(parent)
        return parents

    def crossover(self, parent1: List[Module], parent2: List[Module]) -> Tuple[List[Module], List[Module]]:
        num_modules = len(parent1)
        if random.random() > self.crossover_prob:
            child1 = [Module(
                id=mod.id,
                name=mod.name,
                width_mm=mod.width_mm,
                height_mm=mod.height_mm,
                depth_mm=mod.depth_mm,
                mass_kg=mod.mass_kg,
                thermal_w=mod.thermal_w,
                emc_threshold_v_m=mod.emc_threshold_v_m,
                role=mod.role,
                position_mm=mod.position_mm,
                color=mod.color
            ) for mod in parent1]
            child2 = [Module(
                id=mod.id,
                name=mod.name,
                width_mm=mod.width_mm,
                height_mm=mod.height_mm,
                depth_mm=mod.depth_mm,
                mass_kg=mod.mass_kg,
                thermal_w=mod.thermal_w,
                emc_threshold_v_m=mod.emc_threshold_v_m,
                role=mod.role,
                position_mm=mod.position_mm,
                color=mod.color
            ) for mod in parent2]
            return child1, child2
        k = random.randint(1, num_modules - 1)
        child1 = [None] * num_modules
        child2 = [None] * num_modules
        for i in range(k):
            mod1 = parent1[i]
            mod2 = parent2[i]
            child1[i] = Module(
                id=mod1.id,
                name=mod1.name,
                width_mm=mod1.width_mm,
                height_mm=mod1.height_mm,
                depth_mm=mod1.depth_mm,
                mass_kg=mod1.mass_kg,
                thermal_w=mod1.thermal_w,
                emc_threshold_v_m=mod1.emc_threshold_v_m,
                role=mod1.role,
                position_mm=mod1.position_mm,
                color=mod1.color
            )
            child2[i] = Module(
                id=mod2.id,
                name=mod2.name,
                width_mm=mod2.width_mm,
                height_mm=mod2.height_mm,
                depth_mm=mod2.depth_mm,
                mass_kg=mod2.mass_kg,
                thermal_w=mod2.thermal_w,
                emc_threshold_v_m=mod2.emc_threshold_v_m,
                role=mod2.role,
                position_mm=mod2.position_mm,
                color=mod2.color
            )
        for i in range(k, num_modules):
            mod1 = parent2[i]
            mod2 = parent1[i]
            child1[i] = Module(
                id=mod1.id,
                name=mod1.name,
                width_mm=mod1.width_mm,
                height_mm=mod1.height_mm,
                depth_mm=mod1.depth_mm,
                mass_kg=mod1.mass_kg,
                thermal_w=mod1.thermal_w,
                emc_threshold_v_m=mod1.emc_threshold_v_m,
                role=mod1.role,
                position_mm=mod1.position_mm,
                color=mod1.color
            )
            child2[i] = Module(
                id=mod2.id,
                name=mod2.name,
                width_mm=mod2.width_mm,
                height_mm=mod2.height_mm,
                depth_mm=mod2.depth_mm,
                mass_kg=mod2.mass_kg,
                thermal_w=mod2.thermal_w,
                emc_threshold_v_m=mod2.emc_threshold_v_m,
                role=mod2.role,
                position_mm=mod2.position_mm,
                color=mod2.color
            )
        return child1, child2

    def mutate(self, individual: List[Module]):
        for mod in individual:
            if random.random() < self.mutation_prob:
                dx = random.uniform(-self.mutation_step_mm, self.mutation_step_mm)
                dy = random.uniform(-self.mutation_step_mm, self.mutation_step_mm)
                dz = random.uniform(-self.mutation_step_mm, self.mutation_step_mm)
                x_new = mod.position_mm[0] + dx
                y_new = mod.position_mm[1] + dy
                z_new = mod.position_mm[2] + dz
                half_x = mod.width_mm / 2.0
                half_y = mod.height_mm / 2.0
                half_z = mod.depth_mm / 2.0
                x_new = max(half_x, min(x_new, self.compartment.width_mm - half_x))
                y_new = max(half_y, min(y_new, self.compartment.height_mm - half_y))
                z_new = max(half_z, min(z_new, self.compartment.depth_mm - half_z))
                mod.position_mm = (x_new, y_new, z_new)

    def create_next_generation(self):
        sorted_indices = sorted(range(self.population_size), key=lambda i: self.fitness_values[i], reverse=True)
        new_population: List[List[Module]] = []
        for i in range(self.elitism_size):
            best_idx = sorted_indices[i]
            elite = []
            for mod in self.population[best_idx]:
                cloned = Module(
                    id=mod.id,
                    name=mod.name,
                    width_mm=mod.width_mm,
                    height_mm=mod.height_mm,
                    depth_mm=mod.depth_mm,
                    mass_kg=mod.mass_kg,
                    thermal_w=mod.thermal_w,
                    emc_threshold_v_m=mod.emc_threshold_v_m,
                    role=mod.role,
                    position_mm=mod.position_mm,
                    color=mod.color,
                )
                elite.append(cloned)
            new_population.append(elite)

        parents = self.select_parents()
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
            if len(new_population) >= self.population_size:
                break

        while len(new_population) < self.population_size:
            individual = []
            for template in self.modules_template:
                cloned = Module(
                    id=template.id,
                    name=template.name,
                    width_mm=template.width_mm,
                    height_mm=template.height_mm,
                    depth_mm=template.depth_mm,
                    mass_kg=template.mass_kg,
                    thermal_w=template.thermal_w,
                    emc_threshold_v_m=template.emc_threshold_v_m,
                    role=template.role,
                    position_mm=(0.0, 0.0, 0.0),
                    color=template.color,
                )
                cloned.set_random_position(self.compartment)
                individual.append(cloned)
            new_population.append(individual)

        self.population = new_population

    def run(self) -> Tuple[List[Module], float]:
        self.initialize_population()
        self.evaluate_population()
        best_fitness = 0.0
        best_individual: Optional[List[Module]] = None
        for gen in range(self.generations):
            self.create_next_generation()
            self.evaluate_population()
            current_best_idx = max(range(self.population_size), key=lambda i: self.fitness_values[i])
            current_best_fitness = self.fitness_values[current_best_idx]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = []
                for mod in self.population[current_best_idx]:
                    cloned = Module(
                        id=mod.id,
                        name=mod.name,
                        width_mm=mod.width_mm,
                        height_mm=mod.height_mm,
                        depth_mm=mod.depth_mm,
                        mass_kg=mod.mass_kg,
                        thermal_w=mod.thermal_w,
                        emc_threshold_v_m=mod.emc_threshold_v_m,
                        role=mod.role,
                        position_mm=mod.position_mm,
                        color=mod.color,
                    )
                    best_individual.append(cloned)
            print(f"Поколение {gen + 1}/{self.generations}, лучший fitness = {best_fitness:.6f}")
            if math.isclose(best_fitness, 1.0, abs_tol=1e-6):
                print("Идеальное решение найдено. Остановка ранняя.")
                break
        if best_individual is None:
            best_individual = self.population[0]
            best_fitness = self.fitness_values[0]
        return best_individual, best_fitness

# =============================================================================
# Класс Visualizer3D
# =============================================================================
class Visualizer3D:
    def __init__(self, compartment: Compartment, modules: List[Module]):
        self.compartment = compartment
        self.modules = modules

    def plot(self, show_center_of_mass: bool = True, save_path: Optional[str] = None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D-Визуализация размещения модулей в отсеке", pad=20)
        ax.set_xlim(0, self.compartment.width_mm)
        ax.set_ylim(0, self.compartment.height_mm)
        ax.set_zlim(0, self.compartment.depth_mm)
        ax.set_xlabel("X (мм)")
        ax.set_ylabel("Y (мм)")
        ax.set_zlabel("Z (мм)")
        self._draw_compartment_frame(ax)
        for mod in self.modules:
            mod.update_box_corners()
            self._draw_module(ax, mod)
        if show_center_of_mass:
            self._draw_center_of_mass(ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Изображение сохранено в {save_path}")
        plt.show()

    def _draw_compartment_frame(self, ax):
        w = self.compartment.width_mm
        h = self.compartment.height_mm
        d = self.compartment.depth_mm
        corners = np.array([
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0],
            [0, 0, d],
            [w, 0, d],
            [w, h, d],
            [0, h, d]
        ])
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for e in edges:
            p1 = corners[e[0]]
            p2 = corners[e[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linewidth=1)

    def _draw_module(self, ax, mod: Module):
        corners = mod.box_corners
        faces_indices = [
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 3, 7, 5]
        ]
        faces = []
        for face in faces_indices:
            polygon = [corners[idx] for idx in face]
            faces.append(polygon)
        poly3d = Poly3DCollection(faces, alpha=0.6, facecolor=mod.color)
        ax.add_collection3d(poly3d)
        x_c, y_c, z_c = mod.position_mm
        ax.text(x_c, y_c, z_c, f"{mod.id}", color='black', fontsize=8, ha='center', va='center')

    def _draw_center_of_mass(self, ax):
        x_cm, y_cm, z_cm = FitnessFunction.compute_center_of_mass(self.modules)
        ax.scatter([x_cm], [y_cm], [z_cm], color='red', s=50, label='Центр масс')
        ax.legend()

# =============================================================================
# Класс PlacementEngine с GUI на tkinter
# =============================================================================
class PlacementEngineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Система размещения модулей БПЛА")
        self.modules_template: List[Module] = []
        self.compartment: Optional[Compartment] = None
        self.ga: Optional[GeneticAlgorithm] = None
        self._create_widgets()

    def _create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        # Вкладка 1: Параметры отсека
        self.frame_comp = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_comp, text="Параметры отсека")
        self._build_compartment_tab()
        # Вкладка 2: Параметры модулей
        self.frame_mods = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_mods, text="Параметры модулей")
        self._build_modules_tab()
        # Вкладка 3: Параметры ГА
        self.frame_ga = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_ga, text="Параметры ГА")
        self._build_ga_tab()
        # Вкладка 4: Результаты
        self.frame_res = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_res, text="Результаты")
        self._build_results_tab()

    def _build_compartment_tab(self):
        lbl = ttk.Label(self.frame_comp, text="Введите параметры отсека:")
        lbl.grid(row=0, column=0, columnspan=2, pady=5)
        labels = ["Ширина (мм)", "Высота (мм)", "Глубина (мм)", "Макс. масса (кг)", "Макс. тепловыделение (Вт)",
                  "X0 ЦОМ (мм)", "Y0 ЦОМ (мм)", "Z0 ЦОМ (мм)", "ЭП поле (В/м)"]
        self.comp_vars = {}
        for idx, text in enumerate(labels):
            ttk.Label(self.frame_comp, text=text+":").grid(row=idx+1, column=0, sticky='e', padx=5, pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(self.frame_comp, textvariable=var)
            entry.grid(row=idx+1, column=1, sticky='w', padx=5, pady=2)
            self.comp_vars[text] = var
        btn = ttk.Button(self.frame_comp, text="Сохранить параметры отсека", command=self._save_compartment)
        btn.grid(row=len(labels)+1, column=0, columnspan=2, pady=10)

    def _save_compartment(self):
        try:
            w = float(self.comp_vars["Ширина (мм)"].get())
            h = float(self.comp_vars["Высота (мм)"].get())
            d = float(self.comp_vars["Глубина (мм)"].get())
            m = float(self.comp_vars["Макс. масса (кг)"].get())
            t = float(self.comp_vars["Макс. тепловыделение (Вт)"].get())
            x0 = float(self.comp_vars["X0 ЦОМ (мм)"].get())
            y0 = float(self.comp_vars["Y0 ЦОМ (мм)"].get())
            z0 = float(self.comp_vars["Z0 ЦОМ (мм)"].get())
            ep = float(self.comp_vars["ЭП поле (В/м)"].get())
            self.compartment = Compartment(width_mm=w, height_mm=h, depth_mm=d,
                                           max_mass_kg=m, max_thermal_w=t,
                                           desired_com_mm=(x0, y0, z0), emc_field_v_m=ep)
            messagebox.showinfo("ОК", "Параметры отсека сохранены.")
            self.notebook.select(self.frame_mods)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Некорректные параметры отсека: {e}")

    def _build_modules_tab(self):
        lbl = ttk.Label(self.frame_mods, text="Введите количество модулей:")
        lbl.grid(row=0, column=0, pady=5)
        self.mod_count_var = tk.StringVar()
        entry = ttk.Entry(self.frame_mods, textvariable=self.mod_count_var)
        entry.grid(row=0, column=1, pady=5)
        btn = ttk.Button(self.frame_mods, text="Подтвердить", command=self._init_module_entries)
        btn.grid(row=0, column=2, padx=5)
        self.module_entries_frame = ttk.Frame(self.frame_mods)
        self.module_entries_frame.grid(row=1, column=0, columnspan=3, pady=10)

    def _init_module_entries(self):
        try:
            n = int(self.mod_count_var.get())
            if n <= 0:
                raise ValueError
        except:
            messagebox.showerror("Ошибка", "Количество модулей должно быть положительным целым числом.")
            return
        for widget in self.module_entries_frame.winfo_children():
            widget.destroy()
        self.mod_entries = []
        headers = ["ID", "Наименование", "Ширина", "Высота", "Глубина", "Масса", "Тепло", "Порог ЭМС", "Роль"]
        for j, h in enumerate(headers):
            ttk.Label(self.module_entries_frame, text=h).grid(row=0, column=j, padx=3)
        for i in range(n):
            row_vars = []
            for j in range(len(headers)):
                var = tk.StringVar()
                ent = ttk.Entry(self.module_entries_frame, textvariable=var, width=10)
                ent.grid(row=i+1, column=j, padx=2, pady=2)
                row_vars.append(var)
            self.mod_entries.append(row_vars)
        btn_save = ttk.Button(self.module_entries_frame, text="Сохранить модули", command=self._save_modules)
        btn_save.grid(row=n+1, column=0, columnspan=len(headers), pady=10)

    def _save_modules(self):
        templates = []
        try:
            for row in self.mod_entries:
                mod_id = row[0].get().strip() or row[0].get()
                mod_name = row[1].get().strip() or mod_id
                width = float(row[2].get())
                height = float(row[3].get())
                depth = float(row[4].get())
                mass = float(row[5].get())
                thermal = float(row[6].get())
                emc_thresh = float(row[7].get())
                role = row[8].get().strip().lower()
                if role not in {'emitter', 'receiver', 'neutral'}:
                    raise ValueError(f"Неверная роль '{role}' в модуле {mod_id}.")
                mod = Module(id=mod_id, name=mod_name, width_mm=width,
                              height_mm=height, depth_mm=depth,
                              mass_kg=mass, thermal_w=thermal,
                              emc_threshold_v_m=emc_thresh, role=role)
                templates.append(mod)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Некорректные данные модулей: {e}")
            return
        self.modules_template = templates
        messagebox.showinfo("ОК", "Модули сохранены.")
        self.notebook.select(self.frame_ga)

    def _build_ga_tab(self):
        labels = ["Размер популяции", "Кол-во поколений", "Вероятность скрещивания", "Вероятность мутации", "Размер элитизма", "Мин. ЭМС-разстояние", "Шаг мутации"]
        self.ga_vars = {}
        for i, text in enumerate(labels):
            ttk.Label(self.frame_ga, text=text+":").grid(row=i, column=0, sticky='e', padx=5, pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(self.frame_ga, textvariable=var)
            entry.grid(row=i, column=1, sticky='w', padx=5, pady=2)
            self.ga_vars[text] = var
        btn = ttk.Button(self.frame_ga, text="Сохранить параметры ГА", command=self._save_ga)
        btn.grid(row=len(labels), column=0, columnspan=2, pady=10)

    def _save_ga(self):
        try:
            pop_size = int(self.ga_vars["Размер популяции"].get())
            generations = int(self.ga_vars["Кол-во поколений"].get())
            pc = float(self.ga_vars["Вероятность скрещивания"].get())
            pm = float(self.ga_vars["Вероятность мутации"].get())
            eli = int(self.ga_vars["Размер элитизма"].get())
            min_dist = float(self.ga_vars["Мин. ЭМС-разстояние"].get())
            mstep = float(self.ga_vars["Шаг мутации"].get())
            if not (0 < pc <= 1 and 0 <= pm <= 1):
                raise ValueError
            if eli < 0 or eli > pop_size:
                raise ValueError
        except Exception:
            messagebox.showerror("Ошибка", "Некорректные параметры ГА.")
            return
        self.ga = GeneticAlgorithm(modules_template=self.modules_template,
                                   compartment=self.compartment,
                                   population_size=pop_size,
                                   generations=generations,
                                   crossover_prob=pc,
                                   mutation_prob=pm,
                                   elitism_size=eli,
                                   min_emc_distance_mm=min_dist,
                                   mutation_step_mm=mstep)
        messagebox.showinfo("ОК", "Параметры ГА сохранены.")
        self.notebook.select(self.frame_res)

    def _build_results_tab(self):
        self.run_btn = ttk.Button(self.frame_res, text="Запустить оптимизацию", command=self._run_optimization)
        self.run_btn.pack(pady=10)
        self.results_text = tk.Text(self.frame_res, width=100, height=20)
        self.results_text.pack(padx=5, pady=5)

    def _run_optimization(self):
        if not self.ga or not self.compartment or not self.modules_template:
            messagebox.showerror("Ошибка", "Завершите ввод данных до запуска оптимизации.")
            return
        self.results_text.delete(1.0, tk.END)
        best_individual, best_fitness = self.ga.run()
        self.results_text.insert(tk.END, f"Лучшая приспособленность: {best_fitness:.6f}\n")
        header = f"{'ID':<10}{'X (мм)':>10}{'Y (мм)':>10}{'Z (мм)':>10}{'Масса (кг)':>12}{'Тепло (Вт)':>12}{'ЭМС (В/м)':>12}\n"
        self.results_text.insert(tk.END, header)
        self.results_text.insert(tk.END, '-' * len(header) + '\n')
        total_mass = 0.0
        total_thermal = 0.0
        for mod in best_individual:
            x, y, z = mod.position_mm
            total_mass += mod.mass_kg
            total_thermal += mod.thermal_w
            emc_ok = self.ga.fitness_function.emc_checker.check_emc_field(mod, self.compartment)
            emc_status = "OK" if emc_ok else "Violation"
            self.results_text.insert(tk.END, f"{mod.id:<10}{x:>10.2f}{y:>10.2f}{z:>10.2f}{mod.mass_kg:>12.2f}{mod.thermal_w:>12.2f}{mod.emc_threshold_v_m:>12.2f} -> {emc_status}\n")
        pair_violations = self.ga.fitness_function.emc_checker.check_emc_between_modules(best_individual)
        if pair_violations:
            self.results_text.insert(tk.END, "\nНарушение ЭМС между модулями (emitter-receiver):\n")
            for em_id, rec_id in pair_violations:
                self.results_text.insert(tk.END, f"  {em_id} слишком близко к {rec_id}\n")
        else:
            self.results_text.insert(tk.END, "\nНарушения ЭМС по близости не обнаружены.\n")
        x_cm, y_cm, z_cm = FitnessFunction.compute_center_of_mass(best_individual)
        F_cm = self.ga.fitness_function.compute_center_of_mass_deviation(best_individual, self.compartment.desired_com_mm)
        self.results_text.insert(tk.END, f"\nСуммарная масса: {total_mass:.2f} кг (макс: {self.compartment.max_mass_kg:.2f} кг)\n")
        self.results_text.insert(tk.END, f"Суммарное тепловыделение: {total_thermal:.2f} Вт (макс: {self.compartment.max_thermal_w:.2f} Вт)\n")
        self.results_text.insert(tk.END, f"Фактический ЦОМ: X={x_cm:.2f} мм, Y={y_cm:.2f} мм, Z={z_cm:.2f} мм\n")
        self.results_text.insert(tk.END, f"Отклонение ЦОМ: {F_cm:.2f} мм\n")
        # 3D-визуализация
        btn = ttk.Button(self.frame_res, text="Показать 3D", command=lambda: self._show_3d(best_individual))
        btn.pack(pady=5)
        btn_save = ttk.Button(self.frame_res, text="Сохранить JSON", command=lambda: self._save_json(best_individual))
        btn_save.pack(pady=5)

    def _show_3d(self, best_individual: List[Module]):
        # создаём новое окно для 3D-просмотра
        win = tk.Toplevel(self.root)
        win.title("3D-Визуализация размещения модулей")

        # готовим фигуру и ось без блокирующего plt.show()
        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D-расположение модулей в отсеке", pad=10)
        ax.set_xlim(0, self.compartment.width_mm)
        ax.set_ylim(0, self.compartment.height_mm)
        ax.set_zlim(0, self.compartment.depth_mm)
        ax.set_xlabel("X (мм)")
        ax.set_ylabel("Y (мм)")
        ax.set_zlabel("Z (мм)")

        # готовим визуализатор и рисуем отсек, модули и центр масс
        vis = Visualizer3D(self.compartment, best_individual)
        vis._draw_compartment_frame(ax)
        for mod in best_individual:
            mod.update_box_corners()
            vis._draw_module(ax, mod)
        vis._draw_center_of_mass(ax)

        # встраиваем фигуру в Tkinter с интерактивной панелью
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    def _save_json(self, best_individual: List[Module]):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        data = {
            "compartment": {
                "width_mm": self.compartment.width_mm,
                "height_mm": self.compartment.height_mm,
                "depth_mm": self.compartment.depth_mm,
                "max_mass_kg": self.compartment.max_mass_kg,
                "max_thermal_w": self.compartment.max_thermal_w,
                "desired_com_mm": list(self.compartment.desired_com_mm),
                "emc_field_v_m": self.compartment.emc_field_v_m
            },
            "modules": []
        }
        for mod in best_individual:
            data["modules"].append({
                "id": mod.id,
                "name": mod.name,
                "width_mm": mod.width_mm,
                "height_mm": mod.height_mm,
                "depth_mm": mod.depth_mm,
                "mass_kg": mod.mass_kg,
                "thermal_w": mod.thermal_w,
                "emc_threshold_v_m": mod.emc_threshold_v_m,
                "role": mod.role,
                "position_mm": list(mod.position_mm)
            })
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("ОК", f"JSON сохранён в {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = PlacementEngineGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("Прервано пользователем.")
        sys.exit(0)
    except Exception as e:
        messagebox.showerror("Критическая ошибка", str(e))
        sys.exit(1)
