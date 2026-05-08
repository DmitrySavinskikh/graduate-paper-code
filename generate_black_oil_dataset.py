"""
Generate a machine-learning dataset from many simplified Black-Oil simulations.

Model:
- 2D quarter five-spot waterflood
- dead-oil oil-water Black-Oil model
- constant BHP injector and constant BHP producer
- no water-cut control, no shutoff logic
- one row per simulation in the output CSV
- optional final pressure/saturation fields saved to NPZ

Outputs:
1) <save_prefix>_summary.csv
   One row per simulation: sampled input parameters + final rates/cumulatives/recovery factor.

2) <save_prefix>_final_fields.npz
   Arrays for final pressure and water saturation fields, useful if the neural network target is a field.

Gas production is included as a column for interface compatibility, but it is zero in this dead-oil model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# -----------------------------
# Units
# -----------------------------
DAY = 24.0 * 3600.0
YEAR = 365.25 * DAY
MD = 9.869233e-16  # m^2
CP = 1.0e-3        # Pa*s


@dataclass
class SimConfig:
    """Fixed numerical and geometric settings."""

    Nx: int = 31
    Ny: int = 31
    Lx: float = 500.0       # m
    Ly: float = 500.0       # m
    h: float = 20.0         # m
    total_years: float = 20.0
    dt_days: float = 3.0
    rw: float = 0.10        # m
    skin: float = 0.0


@dataclass
class SimParams:
    """Parameters sampled independently for every simulation."""

    # Rock and initial state
    phi: float
    k_mD: float
    p_init_MPa: float
    Sw_init: float

    # BHP controls
    p_inj_bhp_MPa: float
    p_prod_bhp_MPa: float

    # PVT parameters
    p_ref_MPa: float
    Bw_ref: float
    Bo_ref: float
    cw_1_per_Pa: float
    co_1_per_Pa: float
    mu_w_cP: float
    mu_o_cP: float
    rho_w_sc: float
    rho_o_sc: float

    # Relative permeability parameters
    Swc: float
    Sor: float
    krw0: float
    kro0: float
    nw_exp: float
    no_exp: float


class QuarterFiveSpotSimulator:
    def __init__(self, config: SimConfig, params: SimParams):
        self.cfg = config
        self.par = params

        self.Nx = config.Nx
        self.Ny = config.Ny
        self.N = self.Nx * self.Ny
        self.dx = config.Lx / config.Nx
        self.dy = config.Ly / config.Ny
        self.V = self.dx * self.dy * config.h
        self.cell_indices = np.arange(self.N)

        self.k_abs = params.k_mD * MD
        self.phi = params.phi
        self.p_init = params.p_init_MPa * 1.0e6
        self.p_inj_bhp = params.p_inj_bhp_MPa * 1.0e6
        self.p_prod_bhp = params.p_prod_bhp_MPa * 1.0e6
        self.p_ref = params.p_ref_MPa * 1.0e6
        self.mu_w = params.mu_w_cP * CP
        self.mu_o = params.mu_o_cP * CP

        self.injector_cell = self.cell_id(0, 0)
        self.producer_cell = self.cell_id(self.Nx - 1, self.Ny - 1)

        self.edge_c, self.edge_n, self.edge_Tbase = self._build_edges()

        re = 0.28 * np.sqrt(self.dx * self.dy)
        self.WI = 2.0 * np.pi * self.k_abs * config.h / (np.log(re / config.rw) + config.skin)

    def cell_id(self, i: int, j: int) -> int:
        return i + self.Nx * j

    def _build_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        edge_c: List[int] = []
        edge_n: List[int] = []
        edge_Tbase: List[float] = []

        Tx_base = self.k_abs * self.cfg.h * self.dy / self.dx
        Ty_base = self.k_abs * self.cfg.h * self.dx / self.dy

        for j in range(self.Ny):
            for i in range(self.Nx - 1):
                edge_c.append(self.cell_id(i, j))
                edge_n.append(self.cell_id(i + 1, j))
                edge_Tbase.append(Tx_base)

        for j in range(self.Ny - 1):
            for i in range(self.Nx):
                edge_c.append(self.cell_id(i, j))
                edge_n.append(self.cell_id(i, j + 1))
                edge_Tbase.append(Ty_base)

        return (
            np.asarray(edge_c, dtype=np.int32),
            np.asarray(edge_n, dtype=np.int32),
            np.asarray(edge_Tbase, dtype=float),
        )

    # -----------------------------
    # PVT and flow functions
    # -----------------------------
    def B_w(self, p: np.ndarray | float) -> np.ndarray | float:
        return self.par.Bw_ref / (1.0 + self.par.cw_1_per_Pa * (p - self.p_ref))

    def B_o(self, p: np.ndarray | float) -> np.ndarray | float:
        return self.par.Bo_ref / (1.0 + self.par.co_1_per_Pa * (p - self.p_ref))

    def relperm(self, Sw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        denom = 1.0 - self.par.Swc - self.par.Sor
        Se = (Sw - self.par.Swc) / denom
        Se = np.clip(Se, 0.0, 1.0)
        krw = self.par.krw0 * Se ** self.par.nw_exp
        kro = self.par.kro0 * (1.0 - Se) ** self.par.no_exp
        return krw, kro

    def mobilities(self, Sw: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        krw, kro = self.relperm(Sw)
        lam_w = krw / (self.mu_w * self.B_w(p))
        lam_o = kro / (self.mu_o * self.B_o(p))
        lam_t = lam_w + lam_o
        return lam_w, lam_o, lam_t

    def solve_pressure(self, Sw: np.ndarray, p_old: np.ndarray) -> np.ndarray:
        _, _, lam_t = self.mobilities(Sw, p_old)

        lam_c = lam_t[self.edge_c]
        lam_n = lam_t[self.edge_n]
        T = self.edge_Tbase * (2.0 * lam_c * lam_n / (lam_c + lam_n + 1e-30))

        diag = np.zeros(self.N)
        np.add.at(diag, self.edge_c, T)
        np.add.at(diag, self.edge_n, T)

        b = np.zeros(self.N)

        # Pure water injection endpoint mobility.
        lam_inj = self.par.krw0 / (self.mu_w * self.B_w(self.p_inj_bhp))
        diag[self.injector_cell] += self.WI * lam_inj
        b[self.injector_cell] += self.WI * lam_inj * self.p_inj_bhp

        # BHP producer uses local total mobility.
        diag[self.producer_cell] += self.WI * lam_t[self.producer_cell]
        b[self.producer_cell] += self.WI * lam_t[self.producer_cell] * self.p_prod_bhp

        rows = np.concatenate([self.edge_c, self.edge_n, self.cell_indices])
        cols = np.concatenate([self.edge_n, self.edge_c, self.cell_indices])
        data = np.concatenate([-T, -T, diag])
        A = coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr()
        return spsolve(A, b)

    def saturation_step(self, Sw: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, Dict[str, float]]:
        lam_w, lam_o, lam_t = self.mobilities(Sw, p)
        fw = lam_w / (lam_t + 1e-30)

        lam_c = lam_t[self.edge_c]
        lam_n = lam_t[self.edge_n]
        T = self.edge_Tbase * (2.0 * lam_c * lam_n / (lam_c + lam_n + 1e-30))

        q_total = T * (p[self.edge_c] - p[self.edge_n])
        fw_face = np.where(q_total >= 0.0, fw[self.edge_c], fw[self.edge_n])
        q_water = fw_face * q_total

        div_w = np.zeros(self.N)  # positive when water leaves the cell
        np.add.at(div_w, self.edge_c, q_water)
        np.add.at(div_w, self.edge_n, -q_water)

        # Injector rate, surface m^3/s.
        lam_inj = self.par.krw0 / (self.mu_w * self.B_w(self.p_inj_bhp))
        qwi = max(self.WI * lam_inj * (self.p_inj_bhp - p[self.injector_cell]), 0.0)

        # Producer phase rates, surface m^3/s.
        drawdown = max(p[self.producer_cell] - self.p_prod_bhp, 0.0)
        qwp = self.WI * lam_w[self.producer_cell] * drawdown
        qop = self.WI * lam_o[self.producer_cell] * drawdown
        qgp = 0.0

        source_water = np.zeros(self.N)
        sink_water = np.zeros(self.N)
        source_water[self.injector_cell] = qwi
        sink_water[self.producer_cell] = qwp

        Sw_new = Sw + dt / (self.phi * self.V) * (source_water - sink_water - div_w)
        Sw_new = np.clip(Sw_new, self.par.Swc, 1.0 - self.par.Sor)

        rates = {"qop": qop, "qwp": qwp, "qgp": qgp, "qwi": qwi}
        return Sw_new, rates

    def run(self, keep_final_fields: bool = True) -> Dict[str, object]:
        p = np.full(self.N, self.p_init, dtype=float)
        Sw = np.full(self.N, self.par.Sw_init, dtype=float)

        T_total = self.cfg.total_years * YEAR
        dt = self.cfg.dt_days * DAY
        n_steps = int(np.ceil(T_total / dt))

        OOIP_surface_m3 = np.sum(self.phi * self.V * (1.0 - Sw) / self.B_o(p))

        cum_oil = 0.0
        cum_water = 0.0
        cum_gas = 0.0
        cum_water_inj = 0.0
        last_rates = {"qop": 0.0, "qwp": 0.0, "qgp": 0.0, "qwi": 0.0}

        for step in range(1, n_steps + 1):
            dt_step = min(dt, T_total - (step - 1) * dt)
            if dt_step <= 0.0:
                break

            p = self.solve_pressure(Sw, p)
            Sw, last_rates = self.saturation_step(Sw, p, dt_step)

            cum_oil += last_rates["qop"] * dt_step
            cum_water += last_rates["qwp"] * dt_step
            cum_gas += last_rates["qgp"] * dt_step
            cum_water_inj += last_rates["qwi"] * dt_step

        # Final-time scalar targets.
        result = {
            "final_time_years": self.cfg.total_years,
            "final_oil_prod_rate_m3_day": last_rates["qop"] * DAY,
            "final_water_prod_rate_m3_day": last_rates["qwp"] * DAY,
            "final_gas_prod_rate_m3_day": last_rates["qgp"] * DAY,
            "final_water_inj_rate_m3_day": last_rates["qwi"] * DAY,
            "final_cum_oil_prod_m3": cum_oil,
            "final_cum_water_prod_m3": cum_water,
            "final_cum_gas_prod_m3": cum_gas,
            "final_cum_water_inj_m3": cum_water_inj,
            "final_recovery_factor": cum_oil / OOIP_surface_m3,
            "final_p_min_MPa": float(np.min(p) / 1.0e6),
            "final_p_max_MPa": float(np.max(p) / 1.0e6),
            "final_p_mean_MPa": float(np.mean(p) / 1.0e6),
            "final_Sw_min": float(np.min(Sw)),
            "final_Sw_max": float(np.max(Sw)),
            "final_Sw_mean": float(np.mean(Sw)),
            "final_Sw_producer": float(Sw[self.producer_cell]),
            "OOIP_surface_m3": float(OOIP_surface_m3),
        }

        if keep_final_fields:
            result["p_final"] = p.reshape(self.Ny, self.Nx)
            result["Sw_final"] = Sw.reshape(self.Ny, self.Nx)

        return result


def sample_parameters(rng: np.random.Generator) -> SimParams:
    """Sample one physically reasonable parameter set."""

    # BHPs are sampled so injector > initial pressure > producer in most cases.
    p_prod = rng.uniform(10.0, 16.0)     # MPa
    p_init = rng.uniform(18.0, 22.0)     # MPa
    p_inj = rng.uniform(24.0, 30.0)      # MPa

    return SimParams(
        phi=rng.uniform(0.15, 0.25),
        k_mD=float(np.exp(rng.uniform(np.log(50.0), np.log(300.0)))),
        p_init_MPa=p_init,
        Sw_init=rng.uniform(0.18, 0.30),
        p_inj_bhp_MPa=p_inj,
        p_prod_bhp_MPa=p_prod,
        p_ref_MPa=20.0,
        Bw_ref=rng.uniform(1.00, 1.04),
        Bo_ref=rng.uniform(1.10, 1.45),
        cw_1_per_Pa=rng.uniform(3.0e-10, 6.0e-10),
        co_1_per_Pa=rng.uniform(0.8e-9, 1.8e-9),
        mu_w_cP=rng.uniform(0.4, 0.8),
        mu_o_cP=rng.uniform(1.0, 5.0),
        rho_w_sc=1000.0,
        rho_o_sc=rng.uniform(750.0, 900.0),
        Swc=0.15,
        Sor=0.20,
        krw0=rng.uniform(0.20, 0.40),
        kro0=rng.uniform(0.60, 0.95),
        nw_exp=rng.uniform(1.8, 3.0),
        no_exp=rng.uniform(1.8, 3.0),
    )


def generate_dataset(
    n_simulations: int,
    seed: int = 42,
    config: SimConfig | None = None,
    save_prefix: str = "black_oil_many_sims",
    keep_final_fields: bool = True,
) -> pd.DataFrame:
    """Run many simulations and save a one-row-per-simulation ML dataset."""

    if config is None:
        config = SimConfig()

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    p_fields: List[np.ndarray] = []
    Sw_fields: List[np.ndarray] = []

    for sim_id in range(n_simulations):
        params = sample_parameters(rng)
        sim = QuarterFiveSpotSimulator(config, params)
        out = sim.run(keep_final_fields=keep_final_fields)

        row = {"sim_id": sim_id}
        row.update(asdict(config))
        row.update(asdict(params))

        for key, value in out.items():
            if key not in {"p_final", "Sw_final"}:
                row[key] = value

        rows.append(row)

        if keep_final_fields:
            p_fields.append(out["p_final"])
            Sw_fields.append(out["Sw_final"])

        print(
            f"simulation {sim_id + 1:04d}/{n_simulations:04d} | "
            f"RF={row['final_recovery_factor']:.3f} | "
            f"q_o={row['final_oil_prod_rate_m3_day']:.2f} m3/day"
        )

    df = pd.DataFrame(rows)
    csv_path = f"{save_prefix}_summary.csv"
    df.to_csv(csv_path, index=False)

    if keep_final_fields:
        npz_path = f"{save_prefix}_final_fields.npz"
        np.savez_compressed(
            npz_path,
            p_final_MPa=np.asarray(p_fields) / 1.0e6,
            Sw_final=np.asarray(Sw_fields),
            sim_id=df["sim_id"].to_numpy(),
        )
        print(f"Saved final field arrays to: {npz_path}")

    print(f"Saved summary dataset to: {csv_path}")
    return df


if __name__ == "__main__":
    # Start small to test that everything works.
    # For a real training dataset, increase n_simulations to 100, 500, 1000, etc.
    cfg = SimConfig(
        Nx=31,
        Ny=31,
        total_years=20.0,
        dt_days=3.0,
    )

    dataset = generate_dataset(
        n_simulations=10,
        seed=42,
        config=cfg,
        save_prefix="black_oil_many_sims_demo",
        keep_final_fields=True,
    )

    print(dataset.head())
