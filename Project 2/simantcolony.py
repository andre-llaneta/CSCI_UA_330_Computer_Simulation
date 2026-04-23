"""
Project 2: Agent-Based Simulation of Cordyceps Infection in an Ant Colony
CSCI-UA 330 - Introduction to Computer Simulation
Spring 2026

Simulates an ant colony under normal conditions and then introduces
Ophiocordyceps-like fungal spores. Uses an agent-based stochastic model
where each ant is tracked individually through lifecycle stages.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import os

# ============================================================
# Ant States and Agent Definition
# ============================================================

class AntState(Enum):
    EGG = "egg"
    LARVA = "larva"
    ADULT = "adult"
    INFECTED = "infected"
    ZOMBIE = "zombie"
    SPORULATING = "sporulating"
    DEAD = "dead"


@dataclass
class Ant:
    state: AntState
    days_in_state: int = 0
    age_days: int = 0

    # lifecycle constants
    EGG_DURATION = 14       # days as egg before becoming larva
    LARVA_DURATION = 30     # days as larva before becoming adult

    # cordyceps constants
    ZOMBIE_DURATION = 3     # days as zombie before sporulating
    SPORULATION_DURATION = 7  # days sporulating before spent (dead)

    def is_alive(self):
        return self.state not in (AntState.DEAD, AntState.SPORULATING, AntState.ZOMBIE)

    def consumes_food(self):
        return self.state in (AntState.LARVA, AntState.ADULT, AntState.INFECTED)

    def can_forage(self):
        return self.state in (AntState.ADULT, AntState.INFECTED)

    def advance_day(self):
        self.age_days += 1
        self.days_in_state += 1

    def transition_to(self, new_state: AntState):
        self.state = new_state
        self.days_in_state = 0


# ============================================================
# Colony Simulation
# ============================================================

class ColonySimulation:
    """
    Agent-based simulation of an ant colony with optional cordyceps infection.

    Parameters
    ----------
    n_initial_ants : int
        Number of adult ants at t=0.
    initial_food : float
        Starting food supply in units.
    forage_constant : float
        Constant 'a' in foraging model: each ant brings back Exp(mean=a*N^{-1/3}).
    egg_laying_cap : int
        Maximum eggs per clutch.
    egg_laying_k : float
        Proportionality constant for clutch size = min(k * food/colony_size, cap).
    natural_death_prob : float
        Daily probability of natural death for an adult ant (1/365 default).
    food_cap : float
        Maximum food the colony can store. Excess foraged food is lost.
    cordyceps_enabled : bool
        Whether cordyceps infection is active.
    spore_intro_day : int
        Day on which spores are introduced.
    c0 : float
        Initial spore concentration when introduced.
    infection_rate_p : float
        Infection rate: daily infection probability = c_n * p per foraging ant.
    p_succumb : float
        Daily probability an infected ant succumbs to the parasite.
        Once infected, there is no recovery; the ant either remains
        infected (stalemate) or succumbs on each day.
    spore_birth_rate : float
        Rate 'b' at which sporulating ants add to spore concentration.
    spore_decay_rate : float
        Rate 'k' at which spores decay daily.
    rng_seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_initial_ants=50,
        initial_food=200.0,
        forage_constant=15.0,
        egg_laying_cap=100,
        egg_laying_k=8.0,
        natural_death_prob=1.0/365.0,
        food_cap=500.0,
        continuous_laying=False,
        daily_laying_rate=0.02,
        cordyceps_enabled=False,
        spore_intro_day=200,
        c0=0.05,
        infection_rate_p=0.15,
        p_succumb=0.05,
        spore_birth_rate=0.005,
        spore_decay_rate=0.30,
        rng_seed=None,
    ):
        self.forage_constant = forage_constant
        self.egg_laying_cap = egg_laying_cap
        self.egg_laying_k = egg_laying_k
        self.natural_death_prob = natural_death_prob
        self.food_cap = food_cap
        self.continuous_laying = continuous_laying
        self.daily_laying_rate = daily_laying_rate
        self.cordyceps_enabled = cordyceps_enabled
        self.spore_intro_day = spore_intro_day
        self.c0 = c0
        self.infection_rate_p = infection_rate_p
        self.p_succumb = p_succumb
        self.spore_birth_rate = spore_birth_rate
        self.spore_decay_rate = spore_decay_rate

        self.rng = np.random.default_rng(rng_seed)

        # initialize ants
        self.ants: List[Ant] = []
        for _ in range(n_initial_ants):
            ant = Ant(state=AntState.ADULT)
            ant.age_days = self.rng.integers(30, 200)  # varied starting ages
            self.ants.append(ant)

        self.food_supply = initial_food
        self.spore_concentration = 0.0
        self.day = 0

        # history arrays for plotting
        self.history = {
            "day": [],
            "food": [],
            "n_eggs": [],
            "n_larvae": [],
            "n_adults": [],
            "n_infected": [],
            "n_zombie": [],
            "n_sporulating": [],
            "n_dead": [],
            "n_alive": [],
            "spore_conc": [],
        }

    # ----------------------------------------------------------
    # Census helpers
    # ----------------------------------------------------------

    def _count_state(self, state: AntState) -> int:
        return sum(1 for a in self.ants if a.state == state)

    def _alive_ants(self) -> List[Ant]:
        return [a for a in self.ants if a.is_alive()]

    def _colony_size(self) -> int:
        """Number of living ants (eggs + larvae + adults + infected)."""
        return sum(1 for a in self.ants if a.is_alive())

    def _foraging_ants(self) -> List[Ant]:
        return [a for a in self.ants if a.can_forage()]

    # ----------------------------------------------------------
    # Daily phases
    # ----------------------------------------------------------

    def _phase_lifecycle(self):
        """Advance eggs -> larvae -> adults based on time in state."""
        for ant in self.ants:
            if ant.state == AntState.EGG and ant.days_in_state >= Ant.EGG_DURATION:
                ant.transition_to(AntState.LARVA)
            elif ant.state == AntState.LARVA and ant.days_in_state >= Ant.LARVA_DURATION:
                ant.transition_to(AntState.ADULT)

    def _phase_forage(self):
        """A) Each foraging ant brings back food. Capped at food_cap."""
        foragers = self._foraging_ants()
        n_colony = max(self._colony_size(), 1)
        mean_food = self.forage_constant * (n_colony ** (-1.0 / 3.0))
        for _ in foragers:
            brought = self.rng.exponential(mean_food)
            self.food_supply += brought
        # cap storage
        self.food_supply = min(self.food_supply, self.food_cap)

    def _phase_eat(self):
        """B) Each ant that consumes food eats 1 unit. Unfed ants die."""
        hungry = [a for a in self.ants if a.consumes_food()]
        n_hungry = len(hungry)

        if n_hungry == 0:
            return

        if self.food_supply >= n_hungry:
            # everyone eats
            self.food_supply -= n_hungry
        else:
            # not enough food: randomly choose who eats, rest die
            n_fed = int(self.food_supply)
            self.food_supply = 0.0
            self.rng.shuffle(hungry)
            fed_set = set(id(a) for a in hungry[:n_fed])

            for ant in hungry:
                if id(ant) not in fed_set:
                    ant.transition_to(AntState.DEAD)

    def _phase_natural_death(self):
        """C) Each living ant has a 1/365 chance of dying by natural causes."""
        for ant in self._alive_ants():
            if ant.state == AntState.EGG:
                continue  # eggs don't die of natural causes
            if self.rng.random() < self.natural_death_prob:
                ant.transition_to(AntState.DEAD)

    def _phase_egg_laying(self):
        """D) Queen lays eggs.
        Seasonal mode: twice per year (spring ~day 90, summer ~day 180), spread over 14 days.
        Continuous mode: daily births proportional to number of adults (Peskin-style).
        """
        n_colony = max(self._colony_size(), 1)

        if self.continuous_laying:
            # continuous: expected eggs/day = daily_laying_rate * number of adults
            n_adults = self._count_state(AntState.ADULT) + self._count_state(AntState.INFECTED)
            expected_eggs = self.daily_laying_rate * n_adults
            # only lay if food can support it
            if self.food_supply > n_colony * 2:
                n_eggs = self.rng.poisson(expected_eggs)
                n_eggs = min(n_eggs, 10)  # cap daily eggs
                for _ in range(n_eggs):
                    self.ants.append(Ant(state=AntState.EGG))
            return

        # seasonal mode
        day_of_year = self.day % 365

        # spring laying: days 83-96, summer laying: days 173-186
        spring_window = (83 <= day_of_year <= 96)
        summer_window = (173 <= day_of_year <= 186)

        if not (spring_window or summer_window):
            return

        total_clutch = min(
            round(self.egg_laying_k * self.food_supply / n_colony),
            self.egg_laying_cap
        )
        # spread over 14 days
        eggs_today = max(total_clutch // 14, 1)

        for _ in range(eggs_today):
            self.ants.append(Ant(state=AntState.EGG))

    def _phase_infection(self):
        """Cordyceps infection phase:
        - Foraging ants can get infected by spores.
        - Infected ants play the immune mini-game.
        - Zombies progress to sporulation.
        - Sporulating ants release spores then die.
        """
        if not self.cordyceps_enabled:
            return

        # introduce spores on the intro day
        if self.day == self.spore_intro_day:
            self.spore_concentration = self.c0

        if self.spore_concentration <= 0:
            return

        # infection of foraging ants
        for ant in self._foraging_ants():
            if ant.state == AntState.ADULT:  # not already infected
                p_infect = self.spore_concentration * self.infection_rate_p
                p_infect = min(p_infect, 1.0)  # clamp
                if self.rng.random() < p_infect:
                    ant.transition_to(AntState.INFECTED)

        # immune mini-game for infected ants (no recovery possible)
        for ant in self.ants:
            if ant.state == AntState.INFECTED:
                if self.rng.random() < self.p_succumb:
                    ant.transition_to(AntState.ZOMBIE)  # succumbed

        # zombie -> sporulating
        for ant in self.ants:
            if ant.state == AntState.ZOMBIE and ant.days_in_state >= Ant.ZOMBIE_DURATION:
                ant.transition_to(AntState.SPORULATING)

        # sporulating -> dead (spent)
        for ant in self.ants:
            if ant.state == AntState.SPORULATING and ant.days_in_state >= Ant.SPORULATION_DURATION:
                ant.transition_to(AntState.DEAD)

        # update spore concentration
        n_sporulating = self._count_state(AntState.SPORULATING)
        self.spore_concentration = (
            self.spore_concentration
            + self.spore_birth_rate * n_sporulating
            - self.spore_decay_rate * self.spore_concentration
        )
        self.spore_concentration = max(self.spore_concentration, 0.0)

    # ----------------------------------------------------------
    # Record state
    # ----------------------------------------------------------

    def _record(self):
        self.history["day"].append(self.day)
        self.history["food"].append(self.food_supply)
        self.history["n_eggs"].append(self._count_state(AntState.EGG))
        self.history["n_larvae"].append(self._count_state(AntState.LARVA))
        self.history["n_adults"].append(self._count_state(AntState.ADULT))
        self.history["n_infected"].append(self._count_state(AntState.INFECTED))
        self.history["n_zombie"].append(self._count_state(AntState.ZOMBIE))
        self.history["n_sporulating"].append(self._count_state(AntState.SPORULATING))
        self.history["n_dead"].append(self._count_state(AntState.DEAD))
        self.history["n_alive"].append(self._colony_size())
        self.history["spore_conc"].append(self.spore_concentration)

    # ----------------------------------------------------------
    # Garbage collection: remove dead ants periodically to save memory
    # ----------------------------------------------------------

    def _cleanup_dead(self):
        """Remove dead ants that are no longer sporulating to save memory."""
        self.ants = [a for a in self.ants if a.state != AntState.DEAD]

    # ----------------------------------------------------------
    # Main simulation loop
    # ----------------------------------------------------------

    def run(self, n_days=1825):
        """Run the simulation for n_days."""
        for _ in range(n_days):
            self._record()
            self._phase_lifecycle()
            self._phase_forage()
            self._phase_eat()
            self._phase_natural_death()
            self._phase_egg_laying()
            self._phase_infection()

            # advance all ants by one day
            for ant in self.ants:
                ant.advance_day()

            self.day += 1

            # periodic cleanup every 50 days
            if self.day % 50 == 0:
                self._cleanup_dead()

            # early stop if colony is completely dead
            if self._colony_size() == 0 and self._count_state(AntState.SPORULATING) == 0:
                # record final state and stop
                self._record()
                break

        return self.history


# ============================================================
# Plotting Functions
# ============================================================

def setup_plot_style():
    """Set consistent plot style."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })


def plot_baseline_colony(history, save_path=None):
    """Plot 1: Baseline colony dynamics (no infection) - population + food."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    days = history["day"]

    # population breakdown
    ax1.plot(days, history["n_adults"], label="Adults", color="tab:blue")
    ax1.plot(days, history["n_larvae"], label="Larvae", color="tab:orange")
    ax1.plot(days, history["n_eggs"], label="Eggs", color="tab:green")
    ax1.plot(days, history["n_alive"], label="Total alive", color="black",
             linestyle="--", linewidth=2)
    ax1.set_ylabel("Number of ants")
    ax1.set_title("Baseline Colony Dynamics (No Infection)")
    ax1.legend(loc="best")

    # food supply
    ax2.plot(days, history["food"], color="tab:brown", label="Food supply")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Food units")
    ax2.legend(loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_infection_run(history, save_path=None):
    """Plot 2: Single infection run - population breakdown + spore concentration."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    days = history["day"]

    # stacked population
    ax1.stackplot(
        days,
        history["n_adults"],
        history["n_infected"],
        history["n_zombie"],
        history["n_sporulating"],
        history["n_larvae"],
        history["n_eggs"],
        labels=["Healthy adults", "Infected", "Zombie", "Sporulating", "Larvae", "Eggs"],
        colors=["tab:blue", "tab:red", "tab:purple", "darkred", "tab:orange", "tab:green"],
        alpha=0.8,
    )
    ax1.set_ylabel("Number of ants")
    ax1.set_title("Colony Population Under Cordyceps Infection")
    ax1.legend(loc="upper right", fontsize=9)

    # spore concentration
    ax2.plot(days, history["spore_conc"], color="darkgreen", linewidth=2)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Spore concentration")
    ax2.set_title("Environmental Spore Concentration")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_survival_heatmap(results, c0_values, p_values, save_path=None):
    """Plot 3: Heatmap of colony survival rate vs c0 and infection rate p."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        results, origin="lower", aspect="auto",
        extent=[p_values[0], p_values[-1], c0_values[0], c0_values[-1]],
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    ax.set_xlabel("Infection rate p")
    ax.set_ylabel("Initial spore concentration c₀")
    ax.set_title("Colony Survival Probability (2 years)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Survival rate")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_colony_size_comparison(histories_dict, save_path=None):
    """Plot 4: Colony size comparison for different initial sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"50": "tab:blue", "200": "tab:orange", "1000": "tab:green"}
    for label, runs in histories_dict.items():
        # plot individual runs lightly
        for h in runs:
            ax.plot(h["day"], h["n_alive"], color=colors.get(label, "gray"),
                    alpha=0.15, linewidth=0.8)
        # plot mean
        max_len = max(len(h["day"]) for h in runs)
        # pad shorter runs with 0
        padded = []
        for h in runs:
            arr = np.array(h["n_alive"])
            if len(arr) < max_len:
                arr = np.concatenate([arr, np.zeros(max_len - len(arr))])
            padded.append(arr)
        mean_alive = np.mean(padded, axis=0)
        ax.plot(range(max_len), mean_alive, color=colors.get(label, "gray"),
                linewidth=2.5, label=f"N₀={label} (mean)")

    ax.set_xlabel("Day")
    ax.set_ylabel("Living colony size")
    ax.set_title("Colony Size Resilience Under Cordyceps (multiple runs)")
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_immune_sweep(p_succumb_values, survival_rates, save_path=None):
    """Plot 5: Survival probability vs daily succumb rate (immune strength)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(p_succumb_values, survival_rates, "o-", color="tab:blue",
            markersize=5, linewidth=2)
    ax.set_xlabel("Daily succumb probability (p_succumb)\n← stronger immune | weaker immune →")
    ax.set_ylabel("Colony survival probability at day 730")
    ax.set_title("Effect of Immune Resistance on Colony Survival")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% survival")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stochastic_vs_mean(histories, save_path=None):
    """Plot 6: Multiple stochastic runs overlaid with their mean."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    max_len = max(len(h["day"]) for h in histories)

    # pad and compute mean for n_alive
    padded_alive = []
    padded_infected = []
    for h in histories:
        arr_alive = np.array(h["n_alive"])
        arr_inf = np.array(h["n_infected"])
        if len(arr_alive) < max_len:
            arr_alive = np.concatenate([arr_alive, np.zeros(max_len - len(arr_alive))])
            arr_inf = np.concatenate([arr_inf, np.zeros(max_len - len(arr_inf))])
        padded_alive.append(arr_alive)
        padded_infected.append(arr_inf)

    mean_alive = np.mean(padded_alive, axis=0)
    mean_infected = np.mean(padded_infected, axis=0)
    days = np.arange(max_len)

    # individual runs
    for h in histories:
        ax1.plot(h["day"], h["n_alive"], color="tab:blue", alpha=0.1, linewidth=0.5)
        ax2.plot(h["day"], h["n_infected"], color="tab:red", alpha=0.1, linewidth=0.5)

    ax1.plot(days, mean_alive, color="black", linewidth=2.5, label="Mean of stochastic runs")
    ax1.set_ylabel("Living colony size")
    ax1.set_title(f"Stochastic Variability ({len(histories)} runs)")
    ax1.legend()

    ax2.plot(days, mean_infected, color="black", linewidth=2.5, label="Mean infected")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Number infected")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Parametric Study Runners
# ============================================================

def run_baseline_study(output_dir):
    """Run and plot baseline colony with no infection."""
    print("Running baseline colony simulation (no infection)...")
    sim = ColonySimulation(
        n_initial_ants=50, initial_food=200.0,
        cordyceps_enabled=False, rng_seed=42
    )
    history = sim.run(n_days=730)
    plot_baseline_colony(history, save_path=os.path.join(output_dir, "01_baseline_colony.png"))
    print(f"  Final colony size: {history['n_alive'][-1]}")
    print(f"  Final food supply: {history['food'][-1]:.1f}")


def run_single_infection(output_dir):
    """Run and plot a single infection scenario."""
    print("Running single infection scenario...")
    sim = ColonySimulation(
        n_initial_ants=50, initial_food=200,
        cordyceps_enabled=True, spore_intro_day=200,
        c0=0.05, infection_rate_p=0.18,
        p_succumb=0.05,
        rng_seed=21
    )
    history = sim.run(n_days=1825)
    plot_infection_run(history, save_path=os.path.join(output_dir, "02_single_infection.png"))
    print(f"  Colony alive at end: {history['n_alive'][-1]}")


def run_survival_heatmap(output_dir):
    """Parametric study 1: Vary c0 and p, measure survival."""
    print("Running survival heatmap (c0 vs p)...")
    c0_values = np.linspace(0.01, 0.15, 8)
    p_values = np.linspace(0.05, 0.40, 8)
    n_trials = 10
    n_days = 730

    results = np.zeros((len(c0_values), len(p_values)))

    total = len(c0_values) * len(p_values)
    count = 0

    for i, c0 in enumerate(c0_values):
        for j, p in enumerate(p_values):
            survivals = 0
            for trial in range(n_trials):
                sim = ColonySimulation(
                    n_initial_ants=50, initial_food=200.0,
                    cordyceps_enabled=True, spore_intro_day=200,
                    c0=c0, infection_rate_p=p,
                    p_succumb=0.05,
                    rng_seed=trial * 1000 + i * 100 + j
                )
                history = sim.run(n_days=n_days)
                if history["n_alive"][-1] > 0:
                    survivals += 1
            results[i, j] = survivals / n_trials
            count += 1
            if count % 8 == 0:
                print(f"  Progress: {count}/{total}")

    plot_survival_heatmap(results, c0_values, p_values,
                          save_path=os.path.join(output_dir, "03_survival_heatmap.png"))


def run_colony_size_study(output_dir):
    """Parametric study 2: Vary colony size."""
    print("Running colony size comparison...")
    sizes = [50, 200, 1000]
    n_runs = 8
    n_days = 730
    histories_dict = {}

    for size in sizes:
        print(f"  Colony size {size}...")
        runs = []
        for trial in range(n_runs):
            sim = ColonySimulation(
                n_initial_ants=size, initial_food=size * 4.0,  # scale food with colony
                cordyceps_enabled=True, spore_intro_day=200,
                c0=0.05, infection_rate_p=0.15,
                p_succumb=0.05,
                rng_seed=trial * 100 + size
            )
            history = sim.run(n_days=n_days)
            runs.append(history)
        histories_dict[str(size)] = runs

    plot_colony_size_comparison(histories_dict,
                                save_path=os.path.join(output_dir, "04_colony_size_comparison.png"))


def run_immune_sweep(output_dir):
    """Parametric study 3: Vary immune resistance (p_succumb).
    Lower p_succumb = stronger immune system (slower progression).
    Higher p_succumb = weaker immune system (faster progression).
    """
    print("Running immune strength sweep (varying p_succumb)...")
    # sweep from very slow progression (strong immune) to fast (weak immune)
    p_succumb_values = np.linspace(0.01, 0.15, 12)
    n_trials = 20
    n_days = 730  # 3 years
    survival_rates = []

    for ps in p_succumb_values:
        survivals = 0
        for trial in range(n_trials):
            sim = ColonySimulation(
                n_initial_ants=50, initial_food=200.0,
                cordyceps_enabled=True, spore_intro_day=200,
                c0=0.10, infection_rate_p=0.25,
                p_succumb=ps,
                rng_seed=trial * 100 + int(ps * 10000)
            )
            history = sim.run(n_days=n_days)
            if history["n_alive"][-1] > 0:
                survivals += 1
        survival_rates.append(survivals / n_trials)
        print(f"  p_succumb={ps:.3f} (mean {1/ps:.0f} days to succumb): survival={survival_rates[-1]:.2f}")

    plot_immune_sweep(p_succumb_values, survival_rates,
                      save_path=os.path.join(output_dir, "05_immune_sweep.png"))


def run_stochastic_variability(output_dir):
    """Plot 6: Show stochastic variability across many runs."""
    print("Running stochastic variability study (50 runs)...")
    n_runs = 50
    n_days = 730
    histories = []

    for trial in range(n_runs):
        sim = ColonySimulation(
            n_initial_ants=50, initial_food=200.0,
            cordyceps_enabled=True, spore_intro_day=200,
            c0=0.05, infection_rate_p=0.15,
            p_succumb=0.05,
            rng_seed=trial
        )
        history = sim.run(n_days=n_days)
        histories.append(history)

    plot_stochastic_vs_mean(histories,
                            save_path=os.path.join(output_dir, "06_stochastic_variability.png"))


# ============================================================
# Main
# ============================================================

def plot_prevalence_and_growth(p_values, prevalences, growth_rates, final_sizes, save_path_prev=None, save_path_lambda=None, save_path_size=None):
    """Plots 7-9: prevalence, growth rate, and final size vs infection rate."""

    # Plot 7: Time-averaged prevalence I/N vs p
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, prevalences, "o-", color="tab:red", markersize=5, linewidth=2)
    ax.set_xlabel("Infection rate p")
    ax.set_ylabel("Time-averaged prevalence I/N")
    ax.set_title("Infection prevalence vs infection rate (averaged over 3 years)")
    ax.set_ylim(-0.02, max(max(prevalences) * 1.2, 0.05))
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path_prev:
        plt.savefig(save_path_prev, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 8: Growth rate lambda vs p
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, growth_rates, "o-", color="tab:blue", markersize=5, linewidth=2)
    ax.set_xlabel("Infection rate p")
    ax.set_ylabel("Colony growth rate \u03bb = (1/N)(dN/dt)  [per day]")
    ax.set_title("Colony growth rate vs infection rate")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    plt.tight_layout()
    if save_path_lambda:
        plt.savefig(save_path_lambda, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 9: Final colony size as fraction of baseline
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, final_sizes, "o-", color="tab:green", markersize=5, linewidth=2)
    ax.set_xlabel("Infection rate p")
    ax.set_ylabel("Final colony size / baseline colony size")
    ax.set_title("Colony survival fraction vs infection rate (day 1095)")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(1, color="gray", linestyle="--", alpha=0.3, label="No infection baseline")
    ax.legend()
    plt.tight_layout()
    if save_path_size:
        plt.savefig(save_path_size, dpi=150, bbox_inches="tight")
    plt.close()


def run_prevalence_study(output_dir):
    """Parametric study: vary infection rate p, measure time-averaged prevalence,
    growth rate, and final colony size over 3 years.
    Uses weaker spore dynamics (high k, low b) so infection waves can burn out
    and colonies can potentially survive between seasonal replenishment.
    """
    print("Running prevalence vs infection rate study...")

    p_values = np.concatenate([
        np.linspace(0.0, 0.4, 20),
    ])
    p_values = np.sort(np.unique(p_values))

    n_runs = 20
    n_days = 1095  # 3 years
    n0 = 100
    food0 = 400.0
    food_cap = 600.0

    b = 0.005   # weak spore production
    k = 0.30    # fast decay so waves burn out

    # baseline (no infection)
    print("  Computing no-infection baseline...")
    baseline_sizes = []
    for trial in range(n_runs):
        sim = ColonySimulation(
            n_initial_ants=n0, initial_food=food0,
            food_cap=food_cap,
            cordyceps_enabled=False,
            rng_seed=trial * 9999
        )
        h = sim.run(n_days=n_days)
        baseline_sizes.append(h["n_alive"][-1])
    baseline_mean = max(np.mean(baseline_sizes), 1.0)
    print(f"  Baseline mean colony size at day {n_days}: {baseline_mean:.1f}")

    prevalences = []
    growth_rates = []
    final_sizes = []

    for p in p_values:
        run_prevs = []
        run_lambdas = []
        run_finals = []

        for trial in range(n_runs):
            sim = ColonySimulation(
                n_initial_ants=n0, initial_food=food0,
                food_cap=food_cap,
                cordyceps_enabled=True, spore_intro_day=100,
                c0=0.05, infection_rate_p=p,
                p_succumb=0.05,
                spore_birth_rate=b,
                spore_decay_rate=k,
                rng_seed=trial * 1000 + int(p * 10000)
            )
            history = sim.run(n_days=n_days)

            days = np.array(history["day"])
            n_alive = np.array(history["n_alive"], dtype=float)
            n_infected = np.array(history["n_infected"], dtype=float)

            # time-averaged prevalence over entire post-introduction period
            mask = days >= 100
            if np.sum(mask) < 10:
                run_prevs.append(0.0)
                run_lambdas.append(np.nan)
                run_finals.append(0.0)
                continue

            n_post = n_alive[mask]
            i_post = n_infected[mask]

            with np.errstate(divide='ignore', invalid='ignore'):
                prev_series = np.where(n_post > 0, i_post / n_post, 0.0)
            run_prevs.append(np.mean(prev_series))

            # growth rate: lambda = log(N_end / N_start) / T over last year
            mask2 = days >= 730
            if np.sum(mask2) > 10:
                n_last = n_alive[mask2]
                N_start = n_last[0]
                N_end = n_last[-1]
                T = len(n_last)
                if N_start > 5 and N_end > 0:
                    run_lambdas.append(np.log(max(N_end, 1) / N_start) / T)
                elif N_start > 5 and N_end == 0:
                    run_lambdas.append(-0.01)  # colony died
                else:
                    run_lambdas.append(np.nan)
            else:
                run_lambdas.append(np.nan)

            run_finals.append(n_alive[-1])

        mean_prev = np.nanmean(run_prevs)
        mean_lam = np.nanmean(run_lambdas)
        mean_final = np.mean(run_finals) / baseline_mean
        prevalences.append(mean_prev)
        growth_rates.append(mean_lam if not np.isnan(mean_lam) else 0.0)
        final_sizes.append(mean_final)
        print(f"  p={p:.3f}: prevalence={mean_prev:.4f}, lambda={mean_lam:.6f}, final/baseline={mean_final:.3f}")

    plot_prevalence_and_growth(
        p_values, prevalences, growth_rates, final_sizes,
        save_path_prev=os.path.join(output_dir, "07_prevalence_vs_p.png"),
        save_path_lambda=os.path.join(output_dir, "08_growth_rate_vs_p.png"),
        save_path_size=os.path.join(output_dir, "09_final_size_vs_p.png"),
    )

def main():
    output_dir = "/mnt/user-data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    setup_plot_style()

    # 1) baseline colony validation
#    run_baseline_study(output_dir)

    # 2) single infection run
    run_single_infection(output_dir)

    # 3) survival heatmap (c0 vs p) - skipped for speed
    # run_survival_heatmap(output_dir)

    # 4) colony size comparison
#    run_colony_size_study(output_dir)

    # 5) immune strength sweep
#    run_immune_sweep(output_dir)

    # 6) stochastic variability
#    run_stochastic_variability(output_dir)

    # 7-8) prevalence and growth rate vs infection rate (Peskin-style)
 #   run_prevalence_study(output_dir)

    print("\nAll plots saved to output directory.")
    print("Files: 01_baseline_colony.png through 09_final_size_vs_p.png")


if __name__ == "__main__":
    main()
